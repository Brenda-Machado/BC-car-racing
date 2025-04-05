venv/bin/activate: requirements.txt
	 python3 -m venv venv
	 ./venv/bin/pip install -r requirements.txt	

run: venv/bin/activate
	 ./venv/bin/python3 src/main.py
	
plot: venv/bin/activate
	 ./venv/bin/python3 src/plotreward.py

test: venv/bin/activate
	 ./venv/bin/python3 src/car_racing_v0.py

clean:
	 rm -rf __pycache__
	 rm -rf venv