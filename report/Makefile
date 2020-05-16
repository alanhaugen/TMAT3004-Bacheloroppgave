main.pdf: main.tex sections/method.tex sections/introduction.tex thanks.tex 
	pdflatex -shell-escape main
	biber main
	pdflatex -shell-escape main

clean:
	rm main.pdf
	rm *.aux *.idx *.out *.log *.gz 
	rm sections/*.aux sections/*.idx sections/*.out sections/*.log sections/*.gz 
