.. _troubleshooting:

###############
Troubleshooting
###############


.. _pushing-from-behind:

Pushing when your repo is behind 
=================================

When you try to push new content to the remote copy of the repository::

    git push origin main

and your copy is behind the remote copy (i.e., someone else has pushed new
content to the copy on GitHub since the last time you pulled),
you will get a message that looks something like::

    ! [rejected]        main -> main (fetch first)
    error: failed to push some refs to '/home/jamie/git-fun/local1/../remote'
    hint: Updates were rejected because the remote contains work that you do
    hint: not have locally. This is usually caused by another repository pushing
    hint: to the same ref. You may want to first integrate the remote changes
    hint: (e.g., 'git pull ...') before pushing again.
    hint: See the 'Note about fast-forwards' in 'git push --help' for details.

**OR** a more cryptic message that looks something like (it's more cryptic due
to Git LFS)::

    ref main:: Error in git rev-list --stdin --objects --not --remotes=origin --: exit status 128 fatal: bad object 19dd47de1e8368e425ffbec1a00c8f500f76976a

This is very common, and not a problem at all.
All you need to do is ``pull`` to update your copy::

    git pull origin main

This will likely create a new commit that merges the updates on GitHub with
your new content.
This is common, and Git will create a default message for this merging commit
for you.
After you save and close the commit message, the new merged commit will be
finalized.
Then you should be able to ``push``::

    git push origin main
