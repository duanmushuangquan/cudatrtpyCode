# # 定义c++的编译方式
# cpp_scrs := $(shell find src -name "*.cpp")
# cpp_objs := $(patsubst %.cpp,%.o,$(cpp_scrs))
# cpp_objs := $(subst src,objs,$(cpp_objs))

# # 因为cpp  cu 可能同名但是后缀不一样,所有编译.cu给了一个新的后缀  .cuo(后缀可自定义)
# # 所以在编译阶段建议
# # 定义cuda的编译方式
# cu_scrs := $(shell find src -name "*.cu")
# cu_objs := $(patsubst %.cu,%.cuo,$(cu_scrs))
# cu_objs := $(subst src,objs,$(cu_objs))

# # 定义名称参数
# workspace := workspace
# binary := pro

# #cuda库的目录
# #如果是opencv4，应该是opencv4.5.1/include/opencv4/opencv2
# include_paths := /usr/local/cuda-11.0/include /usr/include/opencv2 \
# 				 /home/tensorrt/TensorRT-8.0.1.6/include/

# #如果是opencv4，应该是opencv4.5.1/lib,青色的是软连接，白色的是最终的文件
# library_paths := /usr/local/cuda-11.0/lib64 /usr/lib/x86_64-linux-gnu \
# 				/home/tensorrt/TensorRT-8.0.1.6/lib

# link_librarys := cudart opencv_core opencv_imgproc opencv_highgui
# # 这个是nvcc   cuda编译默认需要的库名字
# # /usr/local/cuda/lib64/libcudart.so
# # /usr/local/lib/libopencv_core.so
# # /usr/local/lib/libopencv_imgproc.so
# # /usr/local/lib/libopencv_highgui.so

# # 定义编译选项
# cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++11
# cu_compile_flages := -m64 -g -O0 -std=c++11

# # 对头文件路径、库目录统一增加  -I  -L  -l
# rpaths        := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
# include_paths := $(foreach item,$(include_paths),-I$(item))
# library_paths := $(library_paths:%=-L%)
# link_librarys := $(foreach item,$(link_librarys),-l$(item))

# # 合并编译选项
# cpp_compile_flags += $(include_paths)
# cu_compile_flages += $(include_paths)
# link_flags        := $(rpaths) $(link_librarys) $(library_paths) # 动态库链接需要-l -L   -Wl -rpath

# objs/%.o : src/%.cpp
# 	@mkdir -p $(dir $@)
# 	@echo Complie $@
# 	@g++ -c $< -o $@ $(cpp_compile_flags)

# objs/%.cuo : src/%.cu
# 	@mkdir -p $(dir $@)
# 	@echo Complie $@
# 	@nvcc -c $< -o $@ $(cu_compile_flages)

# # 定义workspace/pro文件的编译
# $(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
# 	@mkdir -p $(dir $@)
# 	@echo Link $@
# 	@g++ $^ -o $@ $(link_flags)

# # 定义pro快捷编译指令，这里这里只发生编译，不执行
# pro : $(workspace)/$(binary)

# # 定义编译并执行的指令，并且执行目录切换到workspace下
# run : pro 
# 	@cd $(workspace) && ./$(binary) # 这是一个Shell操作符,允许你根据前一个命令的成功与否来有条件地运行下一个命令。在这里，它确保只有在"cd $(workspace)"命令成功后才执行下一个命令。

# clear:
# 	@rm -rf objs $(workspace)

# debug:
# 	@echo $(rpaths)
# 	@echo $(library_paths)
# 	@echo $(link_librarys)


# .PHONY : debug run pro clear




cc        := g++
name      := pro
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++11
cuda_home := /home/shenlan09/miniconda3/envs/yxymakefile/lib/python3.8/site-packages/trtpy/trt8cuda112cudnn8
syslib    := /home/shenlan09/miniconda3/envs/yxymakefile/lib/python3.8/site-packages/trtpy/lib
cpp_pkg   := /home/shenlan09/miniconda3/envs/yxymakefile/lib/python3.8/site-packages/trtpy/cpp-packages
cuda_arch := 
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)

# 定义cpp的路径查找和依赖项mk文件
cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

# 定义cu文件的路径查找和依赖项mk文件
cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

# 定义opencv和cuda需要用到的库文件
link_cuda      := cudart cudnn
link_trtpro    := 
link_tensorRT  := nvinfer
link_opencv    := 
link_sys       := stdc++ dl
link_librarys  := $(link_cuda) $(link_tensorRT) $(link_sys) $(link_opencv)

# 定义头文件路径，请注意斜杠后边不能有空格
# 只需要写路径，不需要写-I
include_paths := src              \
    $(cuda_home)/include/cuda     \
	$(cuda_home)/include/tensorRT \
	$(cpp_pkg)/opencv4.2/include

# 定义库文件路径，只需要写路径，不需要写-L
library_paths := $(cuda_home)/lib64 $(syslib) $(cpp_pkg)/opencv4.2/lib

# 把library path给拼接为一个字符串，例如a b c => a:b:c
# 然后使得LD_LIBRARY_PATH=a:b:c
empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

# 把库路径和头文件路径拼接起来成一个，批量自动加-I、-L、-l
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
# 如果是 jetson nano，提示找不到-m64指令，请删掉 -m64选项。不影响结果
cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=$(stdcpp) -w -g -O0 -m64 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

# 如果头文件修改了，这里的指令可以让他自动编译依赖的cpp或者cu文件
ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

$(name)   : $(workdir)/$(name)

all       : $(name)
run       : $(name)
	@cd $(workdir) && ./$(name) $(run_args)

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

# 编译cpp依赖项，生成mk文件
$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
# 编译cu文件的依赖项，生成cumk文件
$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

# 定义清理指令
clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/*.trtmodel

# 防止符号被当做文件
.PHONY : clean run $(name)

# 导出依赖库路径，使得能够运行起来
export LD_LIBRARY_PATH:=$(library_path_export)