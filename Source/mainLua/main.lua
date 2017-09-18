#!/usr/bin/env th

require 'torch'
require 'optim'

require 'paths'

require 'xlua'
require 'csvigo'

require 'nn'
require 'dpnn'
-- lua文件是作为一个代码块（chunk）存在的，其实质就是一个函数用dofile调用，代码块就会执行了。
local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

torch.setdefaulttensortype('torch.FloatTensor')

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.device)
end

-- torch.manual_seed(args.seed)#为CPU设置随机种子
-- if cuda:
--    torch.cuda.manual_seed(seed)#为当前GPU设置随机种子
--    torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子

opt.manualSeed = 2
torch.manualSeed(opt.manualSeed)
--########

paths.dofile('dataset.lua')
paths.dofile('batch-represent.lua')

model = torch.load(opt.model)
model:evaluate()
if opt.cuda then
   model:cuda()
end

repsCSV = csvigo.File(paths.concat(opt.outDir, "reps.csv"), 'w')
labelsCSV = csvigo.File(paths.concat(opt.outDir, "labels.csv"), 'w')

batchRepresent()

repsCSV:close()
labelsCSV:close()
