Option Explicit
Dim fso, sh, here
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

' Work from this script's own folder, wherever it was copied to.
here = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = here

' Run hidden (window style 0) and don't wait for it to finish.
sh.Run "cmd /c pixi run python launch.py", 0, False