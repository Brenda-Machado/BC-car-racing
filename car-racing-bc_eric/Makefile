venv/bin/activate: requirements.txt
	 python3 -m venv venv
	 ./venv/bin/pip install -r requirements.txt	

run: venv/bin/activate
	 ./venv/bin/python3 train.py

test: venv/bin/activate
	 ./venv/bin/python3 test.py

ibc: venv/bin/activate
	 ./venv/bin/python3 ibc_trainer.py

test_ibc: venv/bin/activate
	 ./venv/bin/python3 ibc_test.py

clean:
	 rm -rf __pycache__
	 rm -rf venv
