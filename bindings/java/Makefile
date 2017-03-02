swig_dir=Swig
generated_java_dir=$(swig_dir)/com/microsoft/CNTK

ifndef JDK_PATH
    JDK_INCLUDE_PATH=$(JDK_PATH)/include
    JDK_INCLUDE_PATH+=$(JAVA_INCLUDE_PATH)/linux/
    $(info defaulting JAVA_INCLUDE_PATH to $(JAVA_INCLUDE_PATH))
endif

default:
	rm -f $(generated_java_dir)/*.java
	swig -c++ -java -package com.microsoft.CNTK $(INCLUDEPATH:%=-I%) -outdir $(generated_java_dir) cntk_java.i
	javac $(generated_java_dir)/*.java
	java -cvf $(swig_dir)/cntk.java $(swig_dir)/com
	$(CXX) -shared $(COMMON_FLAGS) $(CPPFLAGS) $(CXXFLAGS) -DSWIG $(INCLUDEPATH:%=-I%) $(JDK_INCLUDE_PATH:%=-I%) $(swig_dir)/cntk_java_wrap.cxx -L$(BUILD_TOP)/lib -lcntkmath -lcntklibrary-2.0 -L$(PROTOBUF_PATH)/lib -lprotobuf -o libCNTKJava.so
