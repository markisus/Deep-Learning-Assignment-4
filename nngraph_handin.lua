require 'nn'
require 'nngraph'

summand_size = 10
linear_input_size = 20

x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Identity()()
linear_node = nn.Linear(linear_input_size, summand_size)({x3})
multiply_node = nn.CMulTable()({linear_node, x2})
sum_node = nn.CAddTable()({multiply_node, x1})

m = nn.gModule({x1, x2, x3}, {sum_node})

