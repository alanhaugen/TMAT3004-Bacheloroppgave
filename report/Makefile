main.pdf: main.tex sections/method.tex sections/introduction.tex thanks.tex 
	pdflatex -shell-escape main
	biber main
	pdflatex -shell-escape main
	gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dCompatibilityLevel=1.3 -dPDFSETTINGS=/screen -dEmbedAllFonts=true -dSubsetFonts=true -dColorImageDownsampleType=/Bicubic -dColorImageResolution=144 -dGrayImageDownsampleType=/Bicubic -dGrayImageResolution=144 -dMonoImageDownsampleType=/Bicubic -dMonoImageResolution=144 -sOutputFile=out.pdf main.pdf

clean:
	rm main.pdf
	rm *.aux *.idx *.out *.log *.gz 
	rm sections/*.aux sections/*.idx sections/*.out sections/*.log sections/*.gz 
