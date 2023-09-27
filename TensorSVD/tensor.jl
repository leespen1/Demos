#=
struct TensorIndex{Ndims}
    dims::Vector{Float64}
    index::Vector{Float64}
    function TensorIndex(dims::Vector{Float64}, index::Vector{Float64})
        new{Ndims}(dims, index)
    end
end

"""
Do a integer bit style incrementation. 

Return true when we reach the limit of the indeces
"""
function increment!(ti::TensorIndex{Ndims}) where Ndims
    current_dim = Ndims
    ti.index[current_dim] += 1
    overflow = ti.index[current_dim] > dims[current_dim]

    while (overflow || current_dim < Ndims)
        ti.index[current_dim] = 1
        current_dim += 1
        ti.index[current_dim] += 1
        overflow = ti.index[current_dim] > dims[current_dim]
    end

    return overflow
end

function increment_skip!(ti::TensorIndex{Ndims}, skip_dim) where Ndims
    current_dim = Ndims
    if current_dim != skip_dim
        ti.index[current_dim] += 1
        overflow = ti.index[current_dim] > dims[current_dim]

    end

    while (overflow && current_dim < Ndims)
        if current_dim == skip_dim
            current_dim += 1
            continue
        end
        ti.index[current_dim] = 1
        current_dim += 1
        ti.index[current_dim] += 1
        overflow = ti.index[current_dim] > dims[current_dim]
    end

    return overflow
end

# Maybe this will help
function selectdims(A, dims, indices)
    @assert length(dims) == length(indices) <= ndims(A)
    indexer = repeat(Any[:], ndims(A))
    for (dim, index) in zip(dims, indices)
       indexer[dim] = index
    end
    return A[indexer...]
end

#=
# In this way, I can get the slice I need.
# Another alternative may be to just do the matrix unfolding. Memory
# inefficient, but will work. And I need to do SVD on those matrix unfoldings
# anyway so I should get on it.
#
julia> selectdims(a, (1,2,3), (1,3,:))
3-element Vector{Float64}:
 0.0
 0.0
 0.0

julia> selectdims(a, (1,2), (1,3))
3-element Vector{Float64}:
 0.0
 0.0
 0.0

=#

# selectdim function may help

function matrix_tensor_mult(
        tensor::Array{Float64, N}, matrix::Matrix{Float64}, index::Int64
    ) where N
    matrix_nrows = size(matrix, 1)
    matrix_ncols = size(matrix, 2)
    @assert index <= N
    @assert size(tensor, index) == matrix_ncols

    tensor_size_tup = size(tensor)
    product_size_vec = collect(tensors_size_tup)
    product_size_vec[index] = matrix_nrows # Replace one dimension of the tensor
    product = zeros(product_size_vec...) # Allocate product


    
end
=#

"""
They want 'dim' to be the index which changes most slowly. So under their
notation I should really have a tall matrix unfolding, not a wide one.

But first I'll do it just like they say, for ease of implementation.

Following Definition 1 (unfold_dim = n)
"""
function tensor_unfold(tensor::AbstractArray{T,N}, unfold_dim::Int64) where {T,N}
    @assert 1 <= unfold_dim <= N

    #=
    # Get the lengths of each dimension greater or lesser than the unfold dimension
    greater_dims = collect(unfold_dim+1:N)
    greater_dimlengths = [size(tensor, dim) for dim in greater_dims]

    lesser_dims = collect(1:unfold_dim-1)
    lesser_dimlengths  = [size(tensor, dim) for dim in lesser_dims]

    # Concatenate the two to form the cyclic [I_{n+1}, ... , I_N, I_1, ... , I_{n-1}]
    non_unfold_indices = vcat(greater_dims, lesser_dims)
    non_unfold_dimlengths = vcat(greater_dimlengths, lesser_dimlengths)

    n_cols = reduce(*, greater_dimlengths) * reduce(*, lesser_dimlengths)
    n_rows = size(tensor, unfold_dim)
    =#

    dim_lengths = size(tensor)
    n_rows = dim_lengths[unfold_dim]
    n_cols = div(reduce(*, dim_lengths), n_rows)

    unfolding = Matrix{T}(undef, n_rows, n_cols)

    # Iterate over entries of the tensor, and assign them to the matrix unfolding
    # (maybe it would be more efficient to do it the other way around)
    for multi_index in CartesianIndices(tensor) 
        # Convert to tuple so I can use ':' when indexing the multi-index
        
        row_index = multi_index[unfold_dim]
        col_index = 0
        
        # Do the 'upper' contributions
        for k in 1:N-unfold_dim
            #println("k1: $k")
            # i_{n+1}
            col_index_contrib = multi_index[unfold_dim+k] - 1
            # I_{n+2}*I_{n+3}*...*I_N
            for m in unfold_dim+k+1:N
                #println("m1: $m")
                col_index_contrib *= dim_lengths[m]
            end
            for n in 1:unfold_dim-1
                #println("n1: $n")
                col_index_contrib *= dim_lengths[n]
            end
            col_index += col_index_contrib
        end

        # Do the 'lower contributions'
        for k in 1:unfold_dim-1
            #println("k2: $k")
            col_index_contrib = multi_index[k] - 1
            for m in k+1:unfold_dim-1
                #println("m2: $m")
                col_index_contrib *= dim_lengths[m]
            end
            col_index += col_index_contrib
        end
        # Hanging +1 contribution
        col_index += 1

        # Having obtained the matrix index corresponding to the tensor entry,
        # assign the entry.
        #println("Row index: $row_index\nCol index: $col_index")
        #println("Multi-index: $multi_index")
        unfolding[row_index, col_index] = tensor[multi_index]
    end

    return unfolding
end

function HOSVD(tensor::AbstractArray{T, Ndims}) where {T,Ndims}
    # Gather lext singular matrices of n-mode matrix unfoldings of tensor
    Us = Vector{Matrix{Float64}}(undef, Ndims)
    for n in 1:Ndims
        tensor_unfolded = tensor_unfold(tensor, n)
        svd_obj = svd(tensor_unfolded)
        Us[n] = svd_obj.U
    end

    A1 = tensor_unfold(tensor, 1)
    # Compute 1-mode matrix unfolding of core tensor (1-mode is easiest to program)
    S = Us[1]' *  A1 * reduce(kron, Us[2:end])
    return S, Us
end


"""
dim_lengths are the lengths of each dimension of the tensor.

An arbitrary matrix may be folded into a number of different size tensors, so
this is necessary information.
"""
function tensor_fold(tensor_unfolded::Matrix{T}, unfold_dim::Int64, dim_lengths) where {T}
    N = length(dim_lengths)
    @assert 1 <= unfold_dim <= N

    #=
    # Get the lengths of each dimension greater or lesser than the unfold dimension
    greater_dims = collect(unfold_dim+1:N)
    greater_dimlengths = [size(tensor, dim) for dim in greater_dims]

    lesser_dims = collect(1:unfold_dim-1)
    lesser_dimlengths  = [size(tensor, dim) for dim in lesser_dims]

    # Concatenate the two to form the cyclic [I_{n+1}, ... , I_N, I_1, ... , I_{n-1}]
    non_unfold_indices = vcat(greater_dims, lesser_dims)
    non_unfold_dimlengths = vcat(greater_dimlengths, lesser_dimlengths)

    n_cols = reduce(*, greater_dimlengths) * reduce(*, lesser_dimlengths)
    n_rows = size(tensor, unfold_dim)
    =#

    #dim_lengths = size(tensor)
    n_rows = dim_lengths[unfold_dim]
    n_cols = div(reduce(*, dim_lengths), n_rows)

    tensor = Array{T, N}(undef, dim_lengths...)

    # Iterate over entries of the tensor, and assign them to the matrix unfolding
    # (maybe it would be more efficient to do it the other way around)
    for multi_index in CartesianIndices(tensor) 
        # Convert to tuple so I can use ':' when indexing the multi-index
        
        row_index = multi_index[unfold_dim]
        col_index = 0
        
        # Do the 'upper' contributions
        for k in 1:N-unfold_dim
            #println("k1: $k")
            # i_{n+1}
            col_index_contrib = multi_index[unfold_dim+k] - 1
            # I_{n+2}*I_{n+3}*...*I_N
            for m in unfold_dim+k+1:N
                #println("m1: $m")
                col_index_contrib *= dim_lengths[m]
            end
            for n in 1:unfold_dim-1
                #println("n1: $n")
                col_index_contrib *= dim_lengths[n]
            end
            col_index += col_index_contrib
        end

        # Do the 'lower contributions'
        for k in 1:unfold_dim-1
            #println("k2: $k")
            col_index_contrib = multi_index[k] - 1
            for m in k+1:unfold_dim-1
                #println("m2: $m")
                col_index_contrib *= dim_lengths[m]
            end
            col_index += col_index_contrib
        end
        # Hanging +1 contribution
        col_index += 1

        # Having obtained the matrix index corresponding to the tensor entry,
        # assign the entry.
        #println("Row index: $row_index\nCol index: $col_index")
        #println("Multi-index: $multi_index")
        tensor[multi_index] = tensor_unfolded[row_index, col_index]
    end

    return tensor
end


function example1_tensor()
    A = Array{Int64, 3}(undef, 3, 2, 3)
    A[1,1,1] = A[1,1,2] = A[2,1,1] = 1
    A[2,1,2] = -1
    A[2,1,3] = A[3,1,1] = A[3,1,3] = A[1,2,1] = A[1,2,2] = A[2,2,1] = 2
    A[2,2,2] = -2
    A[2,2,3] = A[3,2,1] = A[3,2,3] = 4
    A[1,1,3] = A[3,1,2] = A[1,2,3] = A[3,2,2] = 0

    return A
end

function example4_tensor()
    tensor_dims = (3,3,3)

    A1 = [
          0.9073  0.7158 -0.3698  1.7842  1.6970  0.0151  2.1236 -0.0740  1.4429
          0.8924 -0.4898  2.4288  1.7753 -1.5077  4.0337 -0.6631  1.9103 -1.7495
          2.1488  0.3054  2.3753  4.2495  0.3207  4.7146  1.8260  2.1335 -0.2716
    ]

    return tensor_fold(A1, 1, tensor_dims)
end


"""
Compare against Section 2, Example 1

Now that the unfolding seems to be correct, once I have the multiplication down
I can use an SVD function from LinearAlgebra to compute the SVD. 
(I should write my own SVD, but I'm short on time)

To test the full SVD, use Example 4 (to 1e-4, or five digit, precision)
"""
function test()
    # Check if matrix unfolding is correct according to example
    A = example1_tensor()

    A1_unfold = [
        1  1  0  2  2  0
        1 -1  2  2 -2  4
        2  0  2  4  0  4
    ]

    @assert A1_unfold == tensor_unfold(A, 1)
    println("Test passed! Example 1 matches!")

    # Check that all matrix unfoldings fold back to the original
    for n in 1:ndims(A)
        unfolded_matrix = tensor_unfold(A, n)
        @assert tensor_fold(unfolded_matrix, n, size(A)) == A
    end
    println("Test passed! Unfoldings fold back into the original!")


    A = example4_tensor() 
    S, Us = HOSVD(A)
    display(S)
    for U in Us
        display(U)
    end

end
