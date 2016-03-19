DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
py.test --cov-report html --cov=audioanalysis -vvv "${DIR}/tests"