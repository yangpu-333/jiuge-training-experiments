"""DP1 state packing: bounded GPU concatenation followed by blocking bulk D2H."""
import torch


def copy_segments(segments, numel, device, chunk_elements=64 * 1024 * 1024):
    """Copy nonoverlapping ranges exactly; fill native-layout holes with zeros."""
    assert numel >= 0 and chunk_elements > 0
    segments = sorted(segments, key=lambda item: item[0])
    previous = 0
    for start, end, source in segments:
        assert previous <= start <= end <= numel
        assert source.dtype == torch.float32 and source.ndim == 1
        assert source.device == device and source.numel() == end - start
        previous = end
    segments = [item for item in segments if item[0] < item[1]]
    output = torch.empty(numel, dtype=torch.float32, device='cpu')
    index = 0
    for lo in range(0, numel, chunk_elements):
        hi = min(numel, lo + chunk_elements)
        while index < len(segments) and segments[index][1] <= lo:
            index += 1
        pieces = []
        position = lo
        cursor = index
        while cursor < len(segments) and segments[cursor][0] < hi:
            start, end, source = segments[cursor]
            left, right = max(lo, start), min(hi, end)
            if position < left:
                pieces.append(torch.zeros(left - position, device=device, dtype=torch.float32))
            if left < right:
                pieces.append(source[left - start:right - start])
                position = right
            cursor += 1
        if position < hi:
            pieces.append(torch.zeros(hi - position, device=device, dtype=torch.float32))
        assert sum(x.numel() for x in pieces) == hi - lo
        merged = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        # No asynchronous CPU read: each whole chunk is finished on return.
        output[lo:hi].copy_(merged, non_blocking=False)
        del merged, pieces
    return output


def pack(optimizer):
    assert optimizer.data_parallel_group.size() == optimizer.data_parallel_group_gloo.size() == 1
    keys = ('param',) + optimizer.optimizer_state_keys
    layouts, devices = [], set()
    for idx, maps in enumerate(optimizer.gbuf_ranges):
        assert len(maps) == 1
        for dtype, buckets in maps.items():
            size = optimizer.buffers[idx].numel_unpadded
            offset = 0
            segments = {key: [] for key in keys}
            for bi, bucket in enumerate(buckets):
                n = optimizer.buffers[idx].buckets[bi].numel_unpadded
                for parameter, ranges in bucket['param_map'].items():
                    lo = offset + ranges['gbuf_local'].start
                    hi = offset + ranges['gbuf_local'].end
                    assert offset <= lo <= hi <= offset + n <= size
                    states = optimizer._get_main_param_and_optimizer_states(parameter)
                    for key in keys:
                        source = states[key].detach()
                        assert source.device.type == 'cuda'
                        devices.add(source.device)
                        segments[key].append((lo, hi, source))
                offset += n
            assert offset == size
            layouts.append((idx, dtype, size, segments))
    assert len(devices) == 1
    device = devices.pop()
    result = {'buckets_coalesced': True}
    with torch.no_grad():
        for idx, dtype, size, segments in layouts:
            tensors = {key: copy_segments(segments[key], size, device) for key in keys}
            tensors['numel_unpadded'] = size
            result[idx] = {dtype: tensors}
    torch.cuda.current_stream(device).synchronize()
    return result


def preflight():
    """Real-device bit-copy cases: holes, boundaries, order, NaNs, signed zeros."""
    patterns = [0, -2147483648, 1065353216, 2143289345, 2139095040, -8388608]
    cases = 0
    assert torch.cuda.device_count() == 4
    for rank in range(4):
        device = torch.device('cuda', rank)
        for length, ranges in [(0, []), (23, [(2, 6), (8, 10), (11, 19)]),
                               (17, [(0, 17)]), (16, [(0, 4), (4, 8), (8, 16)]),
                               (9, []), (12, [(3, 3), (6, 12)])]:
            expected = torch.zeros(length, dtype=torch.float32)
            segments = []
            for lo, hi in ranges:
                values = torch.tensor((patterns * (1 + (hi - lo) // len(patterns)))[:hi - lo], dtype=torch.int32).view(torch.float32)
                expected[lo:hi].copy_(values)
                segments.append((lo, hi, values.to(device)))
            before = [x.cpu().view(torch.int32).clone() for _, _, x in segments]
            for chunk in [1, 5, 64]:
                actual = copy_segments(list(reversed(segments)), length, device, chunk)
                assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                assert all(torch.equal(x.cpu().view(torch.int32), ref) for (_, _, x), ref in zip(segments, before))
                cases += 1
    return {'passed': True, 'cases': cases, 'devices': 4, 'maximum_extra_merged_bytes': 256 * 2**20}
