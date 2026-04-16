all: cp310 cp312

libqzip:
	rm -f qzip/libqzip.so
	make -C qzip install
	mkdir -p lib
	cp qzip/libqzip.so* lib
	cp -a /usr/lib/libqat* lib/
	cp /usr/lib/libusdm_drv_s.so lib/

cp310:
	rm -rf dist/ build/ kv_agent.egg-info/
	CC=gcc CXX=g++ python3.10 setup.py bdist_wheel
	cp dist/*.whl .

cp312:
	rm -rf dist/ build/ kv_agent.egg-info/
	CC=gcc CXX=g++ python3.12 setup.py bdist_wheel
	cp dist/*.whl .
	
clean:
	rm -f *.whl
	rm lib dist/ build/ kv_agent.egg-info/ qzip/libqzip.so qzip/qzip.o -rf
