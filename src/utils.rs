pub unsafe fn slice_as_chunks_unchecked<T, const N: usize>(slice: &[T]) -> &[[T; N]] {
    let chunks =
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const [T; N], slice.len() / N) };
    chunks
}

pub unsafe fn slice_as_chunks_mut_unchecked<T, const N: usize>(slice: &mut [T]) -> &mut [[T; N]] {
    let chunks = unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut [T; N], slice.len() / N)
    };
    chunks
}

pub fn slice_as_chunks<T, const N: usize>(slice: &[T]) -> Option<&[[T; N]]> {
    if slice.len() % N != 0 || slice.len() == 0{
        return None;
    }
    Some(unsafe { slice_as_chunks_unchecked(slice) })
}

pub fn slice_as_chunks_mut<T, const N: usize>(slice: &mut [T]) -> Option<&mut [[T; N]]> {
    if slice.len() % N != 0 || slice.len() == 0{
        return None;
    }
    Some(unsafe { slice_as_chunks_mut_unchecked(slice) })
}
