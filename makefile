run_prototype:
	- cd models  && python sent_lm.py --config ../configs/prototype.rnn_lm_li.yaml 

run_lingala_lm:
	- cd models  && python sent_lm.py --config ../configs/rnn/en_ln/prior.lm_ln.yaml

run_lingala_base:
	- cd models  && python sent_lm.py --config ../configs/rnn/en_ln/rnn.en_ln_base.yaml

run_lingala_ln_fusion:
	- cd models  && python sent_lm.py --config ../configs/rnn/en_ln/rnn.en_ln_fusion.yaml

run_lingala_prior:
	- cd models  && python sent_lm.py --config ../configs/rnn/en_ln/rnn.en_ln_prior.yaml
