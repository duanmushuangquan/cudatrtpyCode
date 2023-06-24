file_name := print.d

ifneq ($(MAKECMDGOALS), clean)
-include $(file_name)
endif

run:
	g++ -M main.cpp -MF main.d

