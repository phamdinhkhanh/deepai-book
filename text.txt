# add the following 2 lines into your .bashrc or .zhsrc (or your choice of bash shell):
# export TABML='/path/to/this/repo'
# alias 2tabml='cd $TABML; source tabml_env/bin/activate; source bashrc'
function tabml_build() {
  cd $TABML
  jupyter-book build book/
  cp -R ./book/data_to_web ./book/_build/html/
}

function tabml_deploy() {
  cd $TABML
  export DEPLOY='_deploy'
  rm -rf $DEPLOY
  mkdir $DEPLOY
  git clone --single-branch --branch gh-pages https://github.com/tiepvupsu/tabml_book $DEPLOY/
  cd $DEPLOY
  rm -Rf *
  cp -r ../book/_build/html/ ./
  git add -f --all .
  DATE_WITH_TIME=`date "+%Y-%m-%d_%H:%M:%S"`
  git commit -m ":rocket: Deploy at $DATE_WITH_TIME"
  git push
  cd ../
  rm -rf $DEPLOY

  # Save cache
  export CACHE='_cache'
  rm -rf $CACHE
  mkdir $CACHE
  git clone --single-branch --branch build https://github.com/tiepvupsu/tabml_book $CACHE/
  cd $CACHE
  rm -Rf *
  cp -r ../book/_build/ ./
  git add -f --all .
  DATE_WITH_TIME=`date "+%Y-%m-%d_%H:%M:%S"`
  git commit -m "Save _build at $DATE_WITH_TIME"
  git push
  cd -
  rm -rf $CACHE
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
