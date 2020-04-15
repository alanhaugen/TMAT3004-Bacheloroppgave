.PHONY: clean

out.avi: inference.py outputs/model_final.pth in.mp4
	python inference.py
	cd .. && ./youtubeuploader_linux_amd64 -headlessAuth -filename TMAT3004-Bacheloroppgave/out.avi

outputs/model_final.pth: train.py data
	python train.py

data:
	curl -L https://www.dropbox.com/s/38ry8ny1lwi1kip/data.zip --output data.zip
	unzip data.zip
	rm data.zip

clean:
	@rm outputs/models_final.pth
	@rm out.avi
