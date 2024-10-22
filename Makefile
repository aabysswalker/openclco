INCLUDE = -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"
LIBDIR = -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64"

%:
	g++ solution$@/main.cpp -o ./solution$@/main $(INCLUDE) $(LIBDIR) -lOpenCL
	./solution$@/main.exe
