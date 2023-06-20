# 定义c++的编译方式
cpp_scrs := $(shell find src -name "*.cpp")
cpp_objs := $(patsubst %.cpp,%.o,$(cpp_scrs))
cpp_objs := $(subst src,objs,$(cpp_objs))

# 因为cpp  cu 可能同名但是后缀不一样,所有编译.cu给了一个新的后缀  .cuo(后缀可自定义)
# 所以在编译阶段建议
# 定义cuda的编译方式
cu_scrs := $(shell find src -name "*.cu")
cu_objs := $(patsubst %.cu,%.cuo,$(cu_scrs))
cu_objs := $(subst src,objs,$(cu_objs))

# 定义名称参数
workspace := workspace
binary := pro

#cuda库的目录
include_paths := /usr/local/cuda-11.0/include /usr/include/opencv2
library_paths := /usr/local/cuda-11.0/lib64 /usr/lib/x86_64-linux-gnu
link_librarys := cudart opencv_core opencv_imgproc opencv_highgui
# 这个是nvcc   cuda编译默认需要的库名字
# /usr/local/cuda/lib64/libcudart.so
# /usr/local/lib/libopencv_core.so
# /usr/local/lib/libopencv_imgproc.so
# /usr/local/lib/libopencv_highgui.so

# 定义编译选项
cpp_compile_flags := -m64 -fPIC -g -O0 -std=c++11
cu_compile_flages := -m64 -g -O0 -std=c++11

# 对头文件路径、库目录统一增加  -I  -L  -l
rpaths        := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(library_paths:%=-L%)
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 合并编译选项
cpp_compile_flags += $(include_paths)
cu_compile_flages += $(include_paths)
link_flags        := $(rpaths) $(link_librarys) $(library_paths) # 动态库链接需要-l -L   -Wl -rpath

objs/%.o : src/%.cpp
	@mkdir -p $(dir $@)
	@echo Complie $@
	@g++ -c $< -o $@ $(cpp_compile_flags)

objs/%.cuo : src/%.cu
	@mkdir -p $(dir $@)
	@echo Complie $@
	@nvcc -c $< -o $@ $(cu_compile_flages)

# 定义workspace/pro文件的编译
$(workspace)/$(binary) : $(cpp_objs) $(cu_objs)
	@mkdir -p $(dir $@)
	@echo Link $@
	@g++ $^ -o $@ $(link_flags)

# 定义pro快捷编译指令，这里这里只发生编译，不执行
pro : $(workspace)/$(binary)

# 定义编译并执行的指令，并且执行目录切换到workspace下
run : pro 
	@cd $(workspace) && ./$(binary) # 这是一个Shell操作符,允许你根据前一个命令的成功与否来有条件地运行下一个命令。在这里，它确保只有在"cd $(workspace)"命令成功后才执行下一个命令。

clear:
	@rm -rf objs $(workspace)

debug:
	@echo $(rpaths)
	@echo $(library_paths)
	@echo $(link_librarys)


.PHONY : debug run pro clear




