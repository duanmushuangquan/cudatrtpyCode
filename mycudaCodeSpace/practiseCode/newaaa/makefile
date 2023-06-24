
cc:= @g++
source_directory := src
object_directory := objs
output_binary_name := pro


sources := $(shell find src -name "*.cpp")
objects := $(patsubst %.cpp,%.o,$(sources))
objects := $(subst $(source_directory)/,$(object_directory)/,$(objects))

compile_flags := -std=c++11
link_flags := 


$(output_binary_name) : $(objects)
	# @mkdir -p $(dir $@)
	@echo Link $@
	$(cc) $^ -o $@

$(object_directory)/%.o : $(source_directory)/%.cpp
	@mkdir -p $(dir $@)
	@echo Compile $<
	$(cc) -c $< -o $@


debug :
	@echo $(sources)
	@echo $(objects)