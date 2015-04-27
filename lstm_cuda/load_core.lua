require 'nngraph'
require 'cunn'
stringx = require('pl.stringx')
require 'io'
require 'base.lua'
data = require 'data.lua'

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  return line
end

--initialize vocab maps (a side effect of loading data)
data.validdataset(1)

core = torch.load('core.net')
g_disable_dropout(core)

-- Initialize start state (empty)
current_state = {}
for i = 1, 4 do current_state[i] = torch.zeros(20, 200):cuda() end

io.write("OK GO\n")
while true do
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    else
      print(line)
      print("Failed, try again")
    end
  else
    entry = data.vocab_map[line]
    if entry == nil then
        entry = data.vocab_map["_"]
    end
    x = torch.Tensor(20):cuda():fill(entry)
    err, new_state, pred = unpack(core:forward({x, x, current_state})) --don't care about label, just put x again
    g_replace_table(current_state, new_state)
    for i = 1, 50 do
    	io.write(pred[1][i].." ")    
    end
    io.write('\n')
  end
end






