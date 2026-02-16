using JuMP
using HiGHS

model = Model(HiGHS.Optimizer)

@variable(model, y[1:3] >= 0)

@objective(model, Min, 6y[1] + 9y[2] + 16y[3])

@constraint(model, y[1] + 2y[2] + 2y[3] >= 6)   
@constraint(model, y[1] + 1y[2] + 3y[3] >= 4)   

optimize!(model)

println("Objective value = ", objective_value(model))
println("y = ", value.(y))
