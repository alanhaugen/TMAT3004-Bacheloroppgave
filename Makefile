.PHONY: clean

out.avi: inference.py outputs/model_final.pth in.mp4
	python inference.py
	cd .. && ./youtubeuploader_linux_amd64 -headlessAuth -filename TMAT3004-Bacheloroppgave/out.avi

outputs/model_final.pth: train.py data
	python train.py

data:
	curl -L https://www.dropbox.com/s/aym2lmnzjlam16v/data.zip --output data.zip
	unzip data.zip
	rm data.zip

clean:
	@rm outputs/models_final.pth
	@rm out.avi
