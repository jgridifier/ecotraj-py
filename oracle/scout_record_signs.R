# Local only (gitignored): rerun the 2026-11-12 m1b increment split with seed 20261112, 999 draws,
# recording every sign vector so the Python sign-flip p-value can be checked exactly.
suppressPackageStartupMessages({library(ecotraj); library(jsonlite)})
D <- "/workspace/scout/1112/data/"; NSIM <- 999
clr <- function(M) { L <- log(M); L - rowMeans(L) }
dirn <- function(C) as.numeric(trajectoryDirectionality(defineTrajectories(dist(C), rep("s", nrow(C)), seq_len(nrow(C)))))
REC <- list()
sflip <- function(C) { d <- diff(C); s <- sample(c(-1, 1), nrow(d), TRUE); REC[[length(REC)+1]] <<- s; rbind(C[1, ], sweep(apply(d * s, 2, cumsum), 2, C[1, ], "+")) }
run <- function(name, mcols) { set.seed(20261112); REC <<- list()
  W <- read.csv(paste0(D, "panel_", name, ".csv"), check.names = FALSE); q <- W$q; M <- as.matrix(W[, -1]); C <- clr(M / rowSums(M)); Tn <- nrow(C)
  Mk <- read.csv(paste0(D, "market_q.csv")); rownames(Mk) <- Mk$observation_date; X <- as.matrix(Mk[q, mcols])
  dC <- diff(C); dX <- diff(X); B <- solve(crossprod(dX), crossprod(dX, dC))
  mk_inc <- dX %*% B; rs_inc <- dC - mk_inc
  mk <- rbind(C[1, ], sweep(apply(mk_inc, 2, cumsum), 2, C[1, ], "+")); rs <- rbind(C[1, ], sweep(apply(rs_inc, 2, cumsum), 2, C[1, ], "+"))
  out <- list()
  for (nm in c("market", "residual")) { P <- if (nm == "market") mk else rs; REC <<- list(); nul <- replicate(NSIM, dirn(sflip(P)))
    out[[nm]] <- list(dir = dirn(P), p_signflip = (1 + sum(nul >= dirn(P))) / (1 + NSIM), null = nul, signs = do.call(rbind, REC)) }
  out }
res <- list(A = run("A", c("vix", "nasdaq_ret4", "baa10y", "ta_ind_g1")), E4 = run("E4", c("vix", "nasdaq_ret4", "baa10y", "eq_rev_ind_g4")))
write_json(res, "m1b_signs.json", digits = NA)
for (p in names(res)) for (nm in c("market","residual")) cat(p, nm, res[[p]][[nm]]$dir, res[[p]][[nm]]$p_signflip, "\n")
