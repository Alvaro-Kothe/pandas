from pandas._libs.dtypes cimport (
    floating_t,
    int64_t,
    numeric_object_t,
    numeric_t,
    uint8_t,
)


cdef numeric_t kth_smallest_c(numeric_t* arr, Py_ssize_t k, Py_ssize_t n) noexcept nogil

cdef enum TiebreakEnumType:
    TIEBREAK_AVERAGE
    TIEBREAK_MIN,
    TIEBREAK_MAX
    TIEBREAK_FIRST
    TIEBREAK_FIRST_DESCENDING
    TIEBREAK_DENSE


cdef numeric_object_t get_rank_nan_fill_val(
    bint rank_nans_highest,
    numeric_object_t val,
    bint is_datetimelike=*,
)


cdef floating_t calc_skew(int64_t minp, int64_t nobs,
                          floating_t mean, floating_t m2, floating_t m3
                          ) noexcept nogil


cdef void compute_moments(
        const floating_t[:] values,
        const Py_ssize_t start_index, const Py_ssize_t size,
        const bint skipna, const uint8_t[:] mask,
        int64_t *nobs_out, floating_t *mean_out, floating_t *m2_out, floating_t *m3_out,
        ) noexcept nogil
