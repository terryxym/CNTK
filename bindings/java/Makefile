swig_dir=Swig
jar_dir=$(swig_dir)/com/microsoft/CNTK

default:
	rm -f $(jar_dir)/*.java
	swig -c++ -java -package com.microsoft.CNTK -I$(SOURCEDIR)/CNTKv2LibraryDll/API -I/home/ratan/CNTK/bindings/common -I$(SOURCEDIR)/Common/Include -outdir com/microsoft/CNTK cntk_java.i
	javac $(jar_dir)/*.java
	jar -cvf $(swig_dir)/cntk.jar $(swig/dir)/com
	$(CXX) -shared -DCPUONLY -DNOSYNC -D_POSIX_SOURCE -D_XOPEN_SOURCE=600 -D__USE_XOPEN2K -std=c++0x -fopenmp -fpermissive -fPIC -Werror -fcheck-new -DSWIG -I$(SOURCEDIR)/Include -I$(SOURCEDIR)/CNTKv2LibraryDll/API -I$(SOURCEDIR)/Common/Include -I/home/ratan/lib/jdk/include -I/home/ratan/lib/jdk/include/linux cntk_java_wrap.cxx -L/home/ratan/CNTK/build/release/lib -lcntkmath -lcntklibrary-2.0 -L/usr/local/protobuf-3.1.0/lib -lprotobuf -o libCNTKJava.so
