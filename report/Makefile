main.pdf: main.tex sections/theory.tex sections/introduction.tex thanks.tex 
	pdflatex main.tex
	bibtex main.tex
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm *.aux *.idx *.out *.log *.gz 
	rm sections/*.aux sections/*.idx sections/*.out sections/*.log sections/*.gz 
