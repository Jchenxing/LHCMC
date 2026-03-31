import numpy as np


def hminired(A):
    m, n = A.shape

    # Subtract column-minimum values from each column.
    col_min = np.min(A, axis=0)
    A -= col_min

    # Subtract row-minimum values from each row.
    row_min = np.min(A, axis=1)
    A -= row_min[:, np.newaxis]

    # Get positions of all zeros.
    zeros_indices = np.argwhere(A == 0)

    # Extend A to give room for row zero list header column.
    A = np.hstack((A, np.zeros((m, 1))))

    for k in range(n):
        i, j = zeros_indices[k]
        # Get all columns in this row.
        cols = np.where(i == zeros_indices[:, 0])[0]
        # Insert pointers in matrix.
        A[i, n] = -j
        A[i, cols] = 0

    return A


def hminiass(A):
    """
    Initial assignment of zeros for the Hungarian method.

    :param A: The reduced cost matrix with linked zeros in each row.
    :return: A tuple containing the modified cost matrix, the cover vector, and the zero list.
    """

    n, np1 = A.shape

    # Initialize return vectors.
    C = np.zeros(n, dtype=int)
    U = np.zeros(n+1, dtype=int)

    # Initialize last/next zero "pointers".
    LZ = np.zeros(n, dtype=int)
    NZ = np.zeros(n, dtype=int)

    for i in range(n):
        # Set j to first unassigned zero in row i.
        lj = n+1
        j = int(-A[i, lj-1])

        # Repeat until we have no more zeros (j==0) or we find a zero
        # in an unassigned column (c(j)==0).
        while C[j-1] != 0:
            # Advance lj and j in zero list.
            lj = j
            j = int(-A[i, lj-1])

            # Stop if we hit end of list.
            if j==0:
                break
        if j != 0:
            # We found a zero in an unassigned column.

            # Assign row i to column j.
            C[j - 1] = i

            # Remove A(i,j) from unassigned zero list.
            A[i, lj - 1] = A[i, j-1]

            # Update next/last unassigned zero pointers.
            NZ[i] = -A[i, j-1]
            LZ[i] = lj-1

            # Indicate A(i,j) is an assigned zero.
            A[i, j-1] = 0
        else:
            # We found no zero in an unassigned column.

            # Check all zeros in this row.

            lj = n+1
            j = -A[i, lj-1]
            while j != 0:
                # Check the row assigned to this column.
                r = C[j - 1]

                # Pick up last/next pointers.
                lm = LZ[r]
                m = NZ[r]

                # Check all unchecked zeros in free list of this row.
                while m != 0:
                    # Stop if we find an unassigned column.
                    if C[m] == 0:
                        break

                    # Advance one step in list.
                    lm = m
                    m = -A[r, lm ]

                if m == 0:
                    # We failed on row r. Continue with next zero on row i.
                    lj = j
                    j = -A[i, lj]
                else:
                    # We found a zero in an unassigned column.

                    # Replace zero at (r,m) in unassigned list with zero at (r,j)
                    A[r, lm] = -j
                    A[r , j ] = A[r, m]

                    # Update last/next pointers in row r.
                    NZ[r ] = -A[r, m]
                    LZ[r] = j

                    # Mark A(r,m) as an assigned zero in the matrix...
                    A[r, m] = 0

                    # ...and in the assignment vector.
                    C[m] = r

                    # Remove A(i,j) from unassigned list.
                    A[i , lj ] = A[i , j ]

                    # Update last/next pointers in row r.
                    NZ[i ] = -A[i , j ]
                    LZ[i] = lj

                    # Mark A(r,m) as an assigned zero in the matrix...
                    A[i, j] = 0

                    # ...and in the assignment vector.
                    C[j] = i

                    # Stop search.
                    break
    r = np.zeros(n)
    rows = C[np.nonzero(C)]  # Find non-zero elements in C
    r[rows - 1] = rows

    empty = np.where(r == 0)[0] + 1  # Find empty rows

    # Create vector with linked list of unassigned rows.
    U = np.zeros(n + 1)
    U[np.concatenate(([n + 1], empty))] = np.concatenate(([empty], [0]))

    return A,C,U


def hmreduce(A, CH, RH, LC, LR, SLC, SLR):
    # Reduce parts of cost matrix in the Hungarian method.

    n = A.shape[0]

    # Find which rows are covered, i.e. unlabelled.
    coveredRows = LR == 0

    # Find which columns are covered, i.e. labelled.
    coveredCols = LC != 0

    r = np.where(~coveredRows)[0]
    c = np.where(~coveredCols)[0]

    # Get minimum of uncovered elements.
    m = np.min(A[r[:, np.newaxis], c])

    # Subtract minimum from all uncovered elements.
    A[r[:, np.newaxis], c] -= m

    # Check all uncovered columns..
    for j in c:
        # ...and uncovered rows in path order..
        for i in SLR:
            # If this is a (new) zero..
            if A[i, j] == 0:
                # If the row is not in unexplored list..
                if RH[i] == 0:
                    # ...insert it first in unexplored list.
                    RH[i] = RH[n]
                    RH[n] = i
                    # Mark this zero as "next free" in this row.
                    CH[i] = j
                # Find last unassigned zero on row I.
                row = A[i, :]
                colsInList = -row[row < 0]
                if len(colsInList) == 0:
                    # No zeros in the list.
                    l = n
                else:
                    l = colsInList[row[colsInList] == 0]
                # Append this zero to end of list.
                A[i, l] = -j

    # Add minimum to all doubly covered elements.
    r = np.where(coveredRows)[0]
    c = np.where(coveredCols)[0]

    # Take care of the zeros we will remove.
    i, j = np.where(A[r[:, np.newaxis], c] <= 0)
    i = r[i]
    j = c[j]

    for k in range(len(i)):
        # Find zero before this in this row.
        lj = np.where(A[i[k], :] == -j[k])[0]
        # Link past it.
        A[i[k], lj] = A[i[k], j[k]]
        # Mark it as assigned.
        A[i[k], j[k]] = 0

    A[r[:, np.newaxis], c] += m

    return A, CH, RH


def hmflip(A, C, LC, LR, U, l, r):
    """
    HMFLIP Flip assignment state of all zeros along a path.

    Parameters:
    A : numpy.ndarray
        The cost matrix.
    C : numpy.ndarray
        The assignment vector.
    LC : numpy.ndarray
        The column label vector.
    LR : numpy.ndarray
        The row label vector.
    U : numpy.ndarray
        The unassigned row list vector.
    l : int
        Position of last zero in path (column index).
    r : int
        Position of last zero in path (row index).

    Returns:
    A : numpy.ndarray
        Updated cost matrix.
    C : numpy.ndarray
        Updated assignment vector.
    U : numpy.ndarray
        Updated unassigned row list vector.
    """

    n = A.shape[0]

    while True:
        # Move assignment in column l to row r.
        C[l] = r

        # Find zero to be removed from zero list.
        # Find zero before this.
        m = np.where(A[r, :] == -l)[0]

        # Link past this zero.
        A[r, m] = A[r, l]
        A[r, l] = 0

        # If this was the first zero of the path.
        if LR[r] < 0:
            # Remove row from unassigned row list and return.
            U[n] = U[r]
            U[r] = 0
            return A, C, U
        else:
            # Move back in this row along the path and get column of next zero.
            l = LR[r]

            # Insert zero at (r,l) first in zero list.
            A[r, l] = A[r, n]
            A[r, n] = -l

            # Continue back along the column to get row of next zero in path.
            r = LC[l]


def hungarian(A):
    m, n = A.shape

    if m != n:
        raise ValueError('Cost matrix must be square!')

    # Save original cost matrix.
    orig = np.copy(A)

    # Reduce matrix.
    A = hminired(A)

    # Do an initial assignment.
    A, C, U = hminiass(A)

    # Repeat while we have unassigned rows.
    while U[n]:
        # Start with no path, no unchecked zeros, and no unexplored rows.
        LR = np.zeros(n, dtype=int)
        LC = np.zeros(n, dtype=int)
        CH = np.zeros(n, dtype=int)
        RH = np.concatenate((np.zeros(n, dtype=int), [-1]))

        # No labelled columns.
        SLC = []

        # Start path in first unassigned row.
        r = U[n]

        # Mark row with end-of-path label.
        LR[r] = -1

        # Insert row first in labelled row set.
        SLR = [r]

        # Repeat until we manage to find an assignable zero.
        while True:
            # If there are free zeros in row r.
            if A[r, n] != 0:
                # Get column of first free zero.
                l = -A[r, n]

                # If there are more free zeros in row r and row r in not yet marked as unexplored.
                if A[r, l] != 0 and RH[r] == 0:
                    # Insert row r first in unexplored list.
                    RH[r] = RH[n]
                    RH[n] = r

                    # Mark in which column the next unexplored zero in this row is.
                    CH[r] = -A[r, l]
            else:
                # If all rows are explored.
                if RH[n] <= 0:
                    # Reduce matrix.
                    A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)

                # Re-start with first unexplored row.
                r = RH[n]

                # Get column of next free zero in row r.
                l = CH[r]

                # Advance "column of next free zero".
                CH[r] = -A[r, l]

                # If this zero is last in the list.
                if A[r, l] == 0:
                    # Remove row r from unexplored list.
                    RH[n] = RH[r]
                    RH[r] = 0

            # While the column l is labelled, i.e. in path.
            while LC[l - 1] != 0:
                # If row r is explored..
                if RH[r - 1] == 0:
                    # If all rows are explored..
                    if RH[n] <= 0:
                        # Reduce cost matrix.
                        A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)

                    # Re-start with first unexplored row.
                    r = RH[n]

                # Get column of next free zero in row r.
                l = CH[r - 1]

                # Advance "column of next free zero".
                CH[r - 1] = -A[r - 1, l - 1]

                # If this zero is last in list..
                if A[r - 1, l - 1] == 0:
                    # ...remove row r from unexplored list.
                    RH[n] = RH[r - 1]
                    RH[r - 1] = 0

                # If the column found is unassigned..
                if C[l - 1] == 0:
                    # Flip all zeros along the path in LR,LC.
                    A, C, U = hmflip(A, C, LC, LR, U, l, r)
                    # ...and exit to continue with next unassigned row.
                    break
                else:
                    # ...else add zero to path.

                    # Label column l with row r.
                    LC[l - 1] = r

                    # Add l to the set of labelled columns.
                    SLC.append(l)

                    # Continue with the row assigned to column l.
                    r = C[l - 1]

                    # Label row r with column l.
                    LR[r - 1] = l

                    # Add r to the set of labelled rows.
                    SLR.append(r)

            # Calculate the total cost.
            T = np.sum(orig[np.logical_and(C, np.arange(1, orig.shape[1] + 1))])

            return C,T