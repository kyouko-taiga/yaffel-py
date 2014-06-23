# This source file is part of yaffel-py
# Main Developer : Dimitri Racordon (kyouko.taiga@gmail.com)
#
# Copyright 2014 Dimitri Racordon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from yaffel.parser import parse
from yaffel.exceptions import EvaluationError
from funcparserlib.parser import NoParseError

import cmd, sys

class Shell(cmd.Cmd):
    intro = 'Yaffel interpreter (version 0.1, June 2014), type Ctrl+D to exit'
    prompt = 'yaffel$ '

    def parse(self, line):
        try:
            t,v = parse(line)
            print('\033[93m[%s]\033[0m %s' % (t.__name__, v))
            return 0
        except NoParseError as e:
            print('\033[91mSyntax error: %s\033[0m' % e)
        except EvaluationError as e:
            print("\033[91mError while evaluating '%s': %s\033[0m" % (line, e))
        except TypeError as e:
            print('\033[91mType inconsistency: %s\033[0m' % e)
        except ZeroDivisionError as e:
            print('\033[91mIllegal operation: Division by zero\033[0m')
        return -1

    def default(self, line):
        if line == 'EOF':
            print('')
            exit(0)
        self.parse(line)

def main():
    shell = Shell()

    # parse the command line input
    if len(sys.argv) > 1 and sys.argv[1] == '-e':
        try:
            exit(shell.default(sys.argv[2]))
        except IndexError:
            print('\033[91musage: shell.py [-e expression]\033[0m') 
            exit(-1)

    Shell().cmdloop()

if __name__ == '__main__':
    main()
