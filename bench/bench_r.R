# Public API black-box benchmark only. Never inspect installed ecotraj source.
args <- commandArgs(trailingOnly=TRUE)
suppressPackageStartupMessages(library(ecotraj))
suppressPackageStartupMessages(library(jsonlite))
stopifnot(as.character(packageVersion('ecotraj')) == '1.2.2')
C <- as.matrix(read.csv(args[1], header=FALSE))
signs <- as.matrix(read.csv(args[2], header=FALSE))
outfile <- args[3]
n <- nrow(C)
stat <- function(path) {
  as.numeric(trajectoryDirectionality(defineTrajectories(
    dist(path), sites=rep('panel', n), surveys=seq_len(n))))
}
inc <- C[-1,,drop=FALSE] - C[-n,,drop=FALSE]
one <- function(s) {
  steps <- sweep(inc, 1, s, '*')
  path <- rbind(C[1,], sweep(apply(steps, 2, cumsum), 2, C[1,], '+'))
  stat(path)
}
t0 <- proc.time()[['elapsed']]
observed <- stat(C)
observed_seconds <- proc.time()[['elapsed']] - t0
# A small pilot chooses whether a 999-draw run is likely to exceed 10 min.
t0 <- proc.time()[['elapsed']]
pilot <- vapply(seq_len(3), function(i) one(signs[i,]), numeric(1))
pilot_seconds <- proc.time()[['elapsed']] - t0
extrapolated <- (n >= 120 && pilot_seconds / 3 * nrow(signs) > 600)
draws <- if (extrapolated) 50L else nrow(signs)
cat(sprintf('R T=%d pilot: %.3fs/draw; measuring %d draws\n', n, pilot_seconds/3, draws))
t0 <- proc.time()[['elapsed']]
null <- vapply(seq_len(draws), function(i) one(signs[i,]), numeric(1))
draw_seconds <- proc.time()[['elapsed']] - t0
result <- list(R=R.version.string, ecotraj=as.character(packageVersion('ecotraj')),
               observed=observed, observed_seconds=observed_seconds,
               measured_draws=draws, measured_draw_seconds=draw_seconds,
               pilot_seconds=pilot_seconds, extrapolated=extrapolated,
               seconds_999=observed_seconds + draw_seconds * nrow(signs) / draws,
               null=null,
               p_value=if (extrapolated) NA_real_ else (1 + sum(null >= observed))/(1 + draws))
write_json(result, outfile, auto_unbox=TRUE, digits=17, pretty=TRUE, na='null')
cat(sprintf('R T=%d: %d draws in %.3fs, projected test %.3fs\n', n, draws, draw_seconds, result$seconds_999))
