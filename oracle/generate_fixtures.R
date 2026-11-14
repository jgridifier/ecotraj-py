#!/usr/bin/env Rscript
# Clean room: public API calls only; never inspect ecotraj implementation.
suppressPackageStartupMessages(library(ecotraj))
library(jsonlite)
stopifnot(as.character(packageVersion("ecotraj")) == "1.2.2")
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
root <- dirname(dirname(normalizePath(script)))
outdir <- file.path(root, "tests", "fixtures")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
metadata <- list(R = R.version.string, ecotraj = as.character(packageVersion("ecotraj")),
                 jsonlite = as.character(packageVersion("jsonlite")), schema_version = 1L)
# Explicit kinds preserve singleton dimensions, labels, mixed data frames and NA.
pack <- function(v) {
  if (inherits(v, "dist")) v <- as.matrix(v)
  if (is.data.frame(v)) return(list(kind = "frame", rows = unname(rownames(v)),
    columns = unname(names(v)), data = lapply(v, function(z) unname(as.vector(z)))))
  if (is.matrix(v)) return(list(kind = "matrix", rows = rownames(v),
    columns = colnames(v), data = unname(v)))
  if (is.list(v)) return(list(kind = "list", data = lapply(v, pack)))
  list(kind = "vector", names = names(v), data = unname(as.vector(v)))
}
set.seed(20261114)
cases <- vector("list", 40L)
for (i in seq_along(cases)) {
  dims <- c(2L, 5L, 14L)[(i - 1L) %% 3L + 1L]
  nsites <- (i - 1L) %% 5L + 1L
  nt <- if (nsites <= 2L) 3L + (i * 7L) %% 10L else 3L + ((i - 1L) %% 4L)
  unequal <- i %% 4L == 0L && nsites > 1L
  counts <- rep(nt, nsites)
  if (unequal) counts[nsites] <- max(3L, nt - 2L)
  if (unequal && length(unique(counts)) == 1L) counts[1L] <- nt + 1L
  sites <- rep(sprintf("site%02d", seq_len(nsites)), counts)
  surveys <- unlist(lapply(counts, seq_len))
  coords <- do.call(rbind, lapply(counts, function(n) {
    z <- matrix(rnorm(n * dims), n, dims)
    sweep(apply(z, 2L, cumsum), 2L, rnorm(dims, sd = 2), "+")
  }))
  metric <- if (i %% 4L == 0L) "squared-euclidean" else if (i %% 7L == 0L) "bray-curtis" else "euclidean"
  D <- as.matrix(dist(coords))
  if (metric == "squared-euclidean") D <- D^2
  if (metric == "bray-curtis") {
    abundance <- exp(coords / 3)
    for (a in seq_len(nrow(D))) for (b in seq_len(ncol(D)))
      D[a,b] <- sum(abs(abundance[a,] - abundance[b,])) / sum(abundance[a,] + abundance[b,])
  }
  times <- if (i %% 3L == 0L) cumsum(runif(max(surveys), .2, 3))[surveys] else NULL
  unsorted <- i %% 2L == 0L
  if (unsorted) {
    order <- sample.int(length(sites))
    D <- D[order, order]; sites <- sites[order]; surveys <- surveys[order]
    if (!is.null(times)) times <- times[order]
  }
  x <- defineTrajectories(D, sites, surveys, times)
  # Intern identical outputs (notably segment endpoint distance matrices).
  pool <- list(); keys <- character(); calls <- list()
  intern <- function(v) {
    if (is.list(v) && !is.data.frame(v)) return(list(kind = "list", data = lapply(v, intern)))
    p <- pack(v)
    key <- as.character(toJSON(p, digits = NA, na = "null", null = "null", auto_unbox = TRUE))
    idx <- match(key, keys)
    if (is.na(idx)) { pool[[length(pool)+1L]] <<- p; keys <<- c(keys, key); idx <- length(pool) }
    list(ref = idx)
  }
  record <- function(rname, pyname, rargs = list(), kwargs = list(), tag = "default") {
    warnings <- character()
    result <- tryCatch(withCallingHandlers({
      value <- do.call(getExportedValue("ecotraj", rname), c(list(x = x), rargs))
      list(status = "ok", value = intern(value))
    }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
      error = function(e) list(status = "error", error = conditionMessage(e)))
    calls[[length(calls)+1L]] <<- c(list(id = paste(pyname, tag, sep = "/"),
      function_name = pyname, kwargs = kwargs, warnings = unique(warnings)), result)
  }
  record("trajectoryLengths", "trajectory_lengths")
  record("trajectoryLengths", "trajectory_lengths", list(relativeToInitial=TRUE), list(relative_to_initial=TRUE), "relative")
  record("trajectoryLengths", "trajectory_lengths", list(all=TRUE), list(all=TRUE), "all")
  record("trajectorySpeeds", "trajectory_speeds")
  for (add in c(TRUE,FALSE)) {
    for (mode in c("default","all","relative")) {
      ra <- list(add=add); pa <- ra
      if (mode == "all") {ra$all <- TRUE; pa$all <- TRUE}
      if (mode == "relative") {ra$relativeToInitial <- TRUE; pa$relative_to_initial <- TRUE}
      record("trajectoryAngles", "trajectory_angles", ra, pa, paste(mode,add,sep="-"))
    }
    record("trajectoryDirectionality", "trajectory_directionality", list(add=add), list(add=add), as.character(add))
  }
  record("trajectoryInternalVariation", "trajectory_internal_variation")
  record("trajectoryMetrics", "trajectory_metrics")
  for (type in c("pairwise.asymmetric","pairwise.symmetric","multiple"))
    record("trajectoryConvergence", "trajectory_convergence", list(type=type), list(type=type), type)
  for (type in c("directed-segment","Hausdorff","PPA"))
    record("segmentDistances", "segment_distances", list(distance.type=type), list(distance_type=type), type)
  for (type in c("DSPD","SPD","Hausdorff")) for (sym in c("mean","max","min","NULL")) {
    sy <- if (sym == "NULL") NULL else sym
    record("trajectoryDistances", "trajectory_distances", list(distance.type=type,symmetrization=sy),
      list(distance_type=type,symmetrization=sy), paste(type,sym,sep="-"))
  }
  violation <- any(vapply(seq_len(nrow(D)), function(k) any(D > outer(D[,k],D[k,],"+") + 1e-10), logical(1)))
  cases[[i]] <- list(id=sprintf("case%02d_%s_%dd_%dsites",i,metric,dims,nsites),
    dimensions=dims, distance_kind=metric, triangle_violation=violation, unequal=unequal,
    unsorted=unsorted, input=list(d=unname(D),sites=unname(sites),surveys=unname(surveys),times=unname(times)),
    outputs=calls, pool=pool)
  cat("Generated", cases[[i]]$id, "\n")
}
write_json(list(metadata=c(metadata,list(seed=20261114L)),cases=cases),
  file.path(outdir,"oracle_cases.json"), digits=I(17), auto_unbox=TRUE, na="null", null="null", pretty=FALSE)

set.seed(20261112)
nsim <- 199L
signcases <- vector("list",5L)
Ts <- c(20L,40L,62L,20L,62L); parts <- c(4L,7L,14L,14L,5L)
for (i in seq_along(signcases)) {
  z <- matrix(rnorm(Ts[i]*parts[i],sd=.3),Ts[i],parts[i])
  z <- sweep(z,2L,rnorm(parts[i],sd=.08),"+")
  C <- apply(z,2L,cumsum)
  C <- C - rowMeans(C) # clr(exp(C)): each row lies in the clr hyperplane.
  d <- diff(C)
  signs <- matrix(NA_integer_,nsim,nrow(d)); null <- numeric(nsim)
  directionality <- function(path) as.numeric(trajectoryDirectionality(
    defineTrajectories(dist(path),rep("clr",nrow(path)),seq_len(nrow(path)))))
  obs <- directionality(C)
  for (j in seq_len(nsim)) {
    s <- sample(c(-1,1), nrow(d), TRUE)
    signs[j,] <- s
    path <- rbind(C[1,], sweep(apply(s*d,2L,cumsum),2L,C[1,],"+"))
    null[j] <- directionality(path)
  }
  signcases[[i]] <- list(id=sprintf("clr%d_T%d_parts%d",i,Ts[i],parts[i]),
    C=unname(C),signs=signs,null=null,observed=obs,p=(1+sum(null>=obs))/(1+nsim))
  cat("Generated",signcases[[i]]$id,"\n")
}
write_json(list(metadata=c(metadata,list(seed=20261112L,nsim=nsim)),cases=signcases),
  file.path(outdir,"signflip_cases.json"), digits=I(17),auto_unbox=TRUE,na="null",null="null",pretty=FALSE)
errors <- unlist(lapply(cases,function(case) vapply(Filter(function(o) o$status=="error",case$outputs),function(o) o$error,character(1))))
cat("\nOracle errors:\n"); print(table(errors))
cat("\nFixture bytes:\n"); print(file.info(file.path(outdir,c("oracle_cases.json","signflip_cases.json")))$size)
stopifnot(file.info(file.path(outdir,"oracle_cases.json"))$size < 5*1024^2)
