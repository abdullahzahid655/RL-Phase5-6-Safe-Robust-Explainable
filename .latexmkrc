# Latexmk configuration for slide.tex
$pdf_previewer = 'evince';
$pdf_mode = 1;
$pdflatex = '/home/test/Desktop/RL_learning_posts/safety_robustness_explainabilty/pdflatex-wrapper';
$latex = 'latex -interaction=nonstopmode';

# Force mode to continue despite errors
$force_mode = 1;

# Ignore errors and continue building
$ignore_error = 1;

# Custom dependency for handling makeindex
sub asy {
    my ($base) = @_;
    system("asy $base.asy");
}


