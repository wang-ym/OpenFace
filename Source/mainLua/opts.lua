local M = { }

-- http://stackoverflow.com/questions/6380820/get-containing-path-of-lua-file
function script_path()
-- 获取当前执行脚本的文件名以及所在路径
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

function M.parse(arg)
-- 提供一种方便解析参数的框架。
   local cmd = torch.CmdLine()
-- text(string)就是输出给定的字符串到屏幕或者文件中
   cmd:text()
   cmd:text('OpenFace')
   cmd:text()
   cmd:text('Options:')

   ------------ General options --------------------
   cmd:option('-outDir', './reps/', 'Subdirectory to output the representations')
   cmd:option('-data',
              paths.concat(script_path(), '..', 'data', 'lfw', 'dlib-affine-sz:96'),
              'Home of dataset')
   cmd:option('-model',
              paths.concat(script_path(), '..', 'models', 'openface', 'nn4.small2.v1.t7'),
              'Path to model to use.')
   cmd:option('-imgDim', 96, 'Image dimension. nn1=224, nn4=96')
   cmd:option('-batchSize',       50,   'mini-batch size')
   cmd:option('-cuda',       false,   'Use cuda')
   cmd:option('-device',       1,   'Cuda device to use')
   cmd:option('-cache',       false,   'Cache loaded data.')
   cmd:text()
   
   local opt = cmd:parse(arg or {})
   os.execute('mkdir -p ' .. opt.outDir)

   return opt
end

return M
