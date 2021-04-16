# add the following 2 lines into your .bashrc or .zhsrc (or your choice of bash shell):
# export DEEPAI='/home/dell/Downloads/deepai-book/'
# alias 2deepai='cd $DEEPAI; source deepai_env/bin/activate; source bashrc'
function deepai_build() {
  cd $DEEPAI
  jupyter-book build book/
  cp -R ./book/data_to_web ./book/_build/html/
}

function convert_notebooks() {
# for every notebook in one folder:
#   convert it to markdown using jupytext
#   prefix it by nb_ to avoid multiple files with same filenames (even with diff exts)

for nb in $(ls *.ipynb)
do
  jupytext $nb --to myst
done

for nb in $(ls *.ipynb)
do
  mv $nb nb_$nb
done
ls
}
