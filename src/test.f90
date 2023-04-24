program testcomms
    use mpi
    use cudafor
    use mod_nvtx
    implicit none
    integer                        :: MPI_ierr, MPI_win, real4_size
    integer                        :: myRank, nRanks
    integer(4)                     :: numIters, iter
    integer(4)                     :: buf_size, i
    integer(KIND=MPI_ADDRESS_KIND) :: size_in_bytes, target_disp
    real(4), allocatable           :: x(:), y(:)

    ! Initialize the MPI environment
    call MPI_Init(MPI_ierr)

    ! Get the number of processes and each mpi rank
    call MPI_Comm_rank(MPI_COMM_WORLD, myRank, MPI_ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nRanks, MPI_ierr)

    if (nRanks .ne. 2) then
        if (myRank .eq. 0) write(*,*) "--ERROR: must be run with 2 ranks!"
        call MPI_Abort(MPI_COMM_WORLD, 1, MPI_ierr)
    end if

    ! Set the size of a real 4
    call MPI_Type_size(MPI_REAL4, real4_size, MPI_ierr)

    ! Set the target displacement to 0
    target_disp = 0

    ! Set how many times the kernel should be ran
    numIters = 10


    ! Set the buffer size for x and y arrays
    buf_size = 512*1000

    ! Set the size in bytes of the buffer
    size_in_bytes = buf_size * real4_size

    ! Allocate the buffers andd pre-initiialize both to 0
    call nvtxRangePush("Allocate buffers")
    allocate(x(buf_size), y(buf_size))
    call nvtxRangePop()

    call nvtxRangePush("Pre-init buffers")
    x(:) = 0.0
    y(:) = 0.0
    call nvtxRangePop()

    ! Form comm windows using x as buffer
    call nvtxRangePush("Form comm windows")
    call MPI_Win_create(x, size_in_bytes, real4_size, MPI_INFO_NULL, MPI_COMM_WORLD, MPI_win, MPI_ierr)
    call nvtxRangePop()

    ! Fence before operations
    call nvtxRangePush("Fence")
    call MPI_Win_fence(0, MPI_win, MPI_ierr)
    call nvtxRangePop()

    ! Iterate
    call nvtxRangePush("Iterate")
    do iter = 1,numIters
        ! Fill the arrays
        call nvtxRangePush("Fill arrays")
        do i = 1,buf_size
            x(i) = myRank + 0.5
            y(i) = 1.5
        end do
        call nvtxRangePop()

        ! Jordi's kernel
        call nvtxRangePush("Jordi's kernel")
        do i = 1,buf_size
            y(i) = 2.0*(x(i)**2) + (y(i)**2) - 2.0*(x(i)**2) + (y(i)**2) + (myRank+1)*1.0
        end do
        call nvtxRangePop()

        ! Rank 0 puts y into window of rank 1
        if (myRank .eq. 0) then
            call nvtxRangePush("Rank 0 put")
            call MPI_Put(y, buf_size, MPI_REAL4, 1, target_disp, buf_size, MPI_REAL4, MPI_win, MPI_ierr)
            call nvtxRangePop()
        end if

        ! Fence after comms
        call nvtxRangePush("Fence")
        call MPI_Win_fence(0, MPI_win, MPI_ierr)
        call nvtxRangePop()

    end do
    call nvtxRangePop()

    ! Finalize thhe MPI environment
    call MPI_Finalize(MPI_ierr)
end program testcomms