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

# Warm both API and path reconstruction before five equal-sized batches.
invisible(one(signs[1,]))
observed_times <- numeric(5)
for (repeat_id in seq_len(5)) {
  observed_times[repeat_id] <- system.time(observed <- stat(C))[['elapsed']]
}
observed_seconds <- median(observed_times)
# Fixed batch sizes replace the noisy 3-draw pilot and its threshold.
draws <- if (n >= 120) 10L else 20L
null <- numeric(draws)
draw_times <- numeric(5)
for (repeat_id in seq_len(5)) {
  t0 <- proc.time()[['elapsed']]
  null <- vapply(seq_len(draws), function(i) one(signs[i,]), numeric(1))
  draw_times[repeat_id] <- proc.time()[['elapsed']] - t0
}
# Prebuild each draw's trajectory outside the bare-call timer.
trajectories <- lapply(seq_len(draws), function(i) {
  steps <- sweep(inc, 1, signs[i,], '*')
  path <- rbind(C[1,], sweep(apply(steps, 2, cumsum), 2, C[1,], '+'))
  defineTrajectories(dist(path), sites=rep('panel', n), surveys=seq_len(n))
})
bare_times <- replicate(5, {
  t0 <- proc.time()[['elapsed']]
  values <- vapply(trajectories, function(x) as.numeric(trajectoryDirectionality(x)), numeric(1))
  stopifnot(max(abs(values - null)) < 1e-12)
  proc.time()[['elapsed']] - t0
})
result <- list(R=R.version.string, ecotraj=as.character(packageVersion('ecotraj')),
               observed=observed, observed_seconds=observed_seconds,
               measured_draws=draws, repeats=5, measured_draw_seconds=median(draw_times),
               draw_samples=draw_times, bare_samples=bare_times,
               bare_seconds_per_draw=median(bare_times)/draws,
               extrapolated=TRUE,
               seconds_999=observed_seconds + median(draw_times) * nrow(signs) / draws,
               null=null, p_value=NA_real_)
write_json(result, outfile, auto_unbox=TRUE, digits=17, pretty=TRUE, na='null')
cat(sprintf('R T=%d: median %d draws %.3fs, projected test %.3fs, bare %.6fs/draw\n',
            n, draws, median(draw_times), result$seconds_999, result$bare_seconds_per_draw))
