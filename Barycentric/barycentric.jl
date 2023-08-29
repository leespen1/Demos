using GLMakie

function draw_polygon!(ax, vertices; draw_vertices=true)
    n_vertices = length(vertices)
    @assert n_vertices > 2

    # Draw the edges
    vertices_looped = copy(vertices) # If I'm drawing a lot of triangles, would be bad to make a copy like this
    push!(vertices_looped, vertices[1])
    lines!(ax, vertices_looped) 

    # Draw the vertices
    if draw_vertices
        scatter!(ax, vertices) # Plot the vertices
    end

    return
end

function barycentric_to_cartesian(bary_coords, vertices)
    n_vertices = length(vertices)
    @assert n_vertices > 2
    @assert n_vertices == length(bary_coords)
    cartesian_coords = [0.0, 0.0]
    bary_coords_scale = sum(bary_coords) # For normalizing barycentric coords
    bary_coords_scaled = bary_coords ./ bary_coords_scale
    display(bary_coords_scale)
    display(bary_coords_scaled)
    for i in eachindex(vertices)
        cartesian_coords .+= bary_coords_scaled[i] .* vertices[i]
    end
    return Point2f(cartesian_coords)
end

fig = Figure()

ax = Axis(fig[1, 1])

sl_a = Slider(fig[2, 1], range = 0:0.01:10, startvalue = 0)
sl_b = Slider(fig[3, 1], range = 0:0.01:10, startvalue = 0)
sl_c = Slider(fig[4, 1], range = 0:0.01:10, startvalue = 0)

vertices = [Point2f(0.0, 0.0), Point2f(0.0, 1.0), Point2f(1.0, 0.0)]

draw_polygon!(ax, vertices)

# When sliders are changed, update barycentric coordinates
point = lift(sl_a.value, sl_a.value, sl_c.value) do a, b, c
    barycentric_to_cartesian([a,b,c], vertices)
end

scatter!(point, color = :red, markersize = 20)

fig
