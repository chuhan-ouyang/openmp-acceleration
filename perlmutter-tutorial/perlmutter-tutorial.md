# Getting Started with Perlmutter

## 2FA and Log-In

You can set up the two-factor authentication (2FA) and log into Perlmutter following the intructions [here](https://docs.nersc.gov/connect/mfa/).
You can access Perlmutter server by following the instruction [here](https://docs.nersc.gov/systems/perlmutter/).
Be sure to set up the `sshproxy` following the directions on that page to avoid typing your 2FA every time you log in. 

## Command Line

Once logged into Perlmutter, you are be presented with a [bash shell](https://en.wikipedia.org/wiki/Bash_(Unix_shell)).

Some important commands are:

```
ls                          # list files in currect directory
ls *.txt                    # list files in currect directory that end in .txt
cd [DIRNAME]                # change current directory
cd ~                        # change to your home directory
cd ..                       # go up one directory
cat [FILENAME]              # print file content
cat [FILENAME] | less       # print file content, scroll with arrows, quit with q
grep "needle" [FILENAME]    # searches for "needle" in file
man [COMMAND]               # display the manual for COMMAND, use q to quit
exit                        # close shell
```

You can go back to the last used command by using `↑` key.
Press `Enter` if you want to use the command, use `Backspace` to edit the
command before using it.
Use `Ctrl+R` to search through your command history.
Use `cat ~/.bash_history | less` to view your complete command history.

## Editor

One of the easiest ways to implement your homework is to directly change
the code on the server.
For this you need to use a command line editor like `vim` (https://en.wikipedia.org/wiki/Vim_(text_editor)) which is loaded by default:

```.sh
student@login04:~/hw1> vim dgemm-blocked.c
```

Use `Esc` and `:q` to exit.
(`:q!` if you want to discard changes).
Try out the [interactive vim tutorial](https://www.openvim.com/) to learn more.

## Compiler

Compilers on Perlmutter are selected using a "programming environment" module.  No matter
which compiler you use, you always compile on Perlmutter using the `CC` (for C++) or `cc`
commands, which use special wrappers that invoke the compilers for you with some
specially tuned flags enabled (such as `-march=native`).

Unlike Cori using Intel compilers by default, Perlmutter uses GNU by default.
You don't have to switch to using the GNU compiler as previous years. To check the loaded modules, 
you can use command `module list`. Now, when you invoke `CC` or `cc`, you will be using the GNU compilers.

Additionally, most homeworks require cmake version > 3.14, which can be loaded using the command `module load cmake`.
You can confirm your module configuration is correct by running `module list`.
For more details on modules, see the documentation on modules [here](https://docs.nersc.gov/environment/modules/).

## Compiler Explorer

You can examine the assembly produced for your code by using the `-S` flag in your compiler.

```Bash
[demmel@blasboss ~] cc -S hello.c -o hello.s
[demmel@blasboss ~]$ head hello.s
        .file   "hello.c"
        .text
        .section        .rodata
.LC0:
        .string "Hi, fam!"
```

While this can be useful, it's often cumbersome to constantly edit a sourcefile and then
have to recompile to view the assembly.  There's a nice tool called the
[GodBolt Compiler Explorer](https://godbolt.org/) that let's you interactively edit
code and view the assembly interactively as you optimize.  It also has a couple of
nice features that you won't find just using your local C/C++ compiler: it will color
code lines of code to visually match them up with the corresponding lines of assembly,
and it will use some advanced symbol lookup techniques to place more human-readable names
for your functions and variables in the assembly output.

## Acknowledgements

Thanks to Kevin Laeufer to create this tutorial.


