<p align="center">
  <img src="https://github.com/DavideStenner/Kaggle/blob/master/Connect%20X/image.png" />
</p>

This challenge consists on definining an agent which can play connect 4.
I used Negamax with alpha beta pruning + Bitboard (to increase speed).

My solution is based on the approach contained in these two link:

- http://blog.gamesolver.org/
- https://towardsdatascience.com/creating-the-perfect-connect-four-ai-bot-c165115557b0

The heuristic function is based on the concept of threats and winning row and is taken from:

- https://pdfs.semanticscholar.org/35de/5f75444e1fc1c96cb6378393f3ff24a74099.pdf
