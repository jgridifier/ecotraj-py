#!/usr/bin/env Rscript
# Clean-room oracle: only public ecotraj calls on synthetic data.
# Default: 30 shuffled cases plus sorted controls and edge cases.
# Optional arguments: output path, number of shuffled cases, RNG seed.
suppressPackageStartupMessages(library(ecotraj))
library(jsonlite)
stopifnot(as.character(packageVersion("ecotraj")) == "1.2.2")
argv <- commandArgs(FALSE)
script <- sub("^--file=", "", argv[grepl("^--file=", argv)])
root <- dirname(dirname(normalizePath(script)))
args <- commandArgs(TRUE)
output <- if (length(args) >= 1) args[1] else file.path(root, "tests/fixtures/convergence_roworder_cases.json")
count <- if (length(args) >= 2) as.integer(args[2]) else 30L
seed <- if (length(args) >= 3) as.integer(args[3]) else 871239L
set.seed(seed)
pack <- function(v) {
  if (is.list(v)) return(list(kind="list", data=lapply(v, pack)))
  list(kind="vector", names=names(v), data=unname(as.vector(v)))
}
cases <- list()
record <- function(id, D, sites, surveys, times, info=list()) {
  warnings <- character()
  stage <- "define"
  result <- tryCatch(withCallingHandlers({
    x <- defineTrajectories(D, sites, surveys, times)
    stage <- "convergence"
    list(status="ok", value=pack(trajectoryConvergence(x, type="multiple")))
  }, warning=function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error=function(e) list(status="error", stage=stage, error=conditionMessage(e)))
  cases[[length(cases)+1L]] <<- c(list(id=id,
    input=list(d=unname(D), sites=sites, surveys=surveys, times=times),
    info=info, warnings=warnings), result)
}
for (i in seq_len(count)) {
  ns <- 2L + (i-1L) %% 4L
  nt <- sample(3:8, 1)
  dims <- c(2L,5L)[1L + (i-1L) %% 2L]
  n <- ns*nt
  C <- matrix(rnorm(n*dims), n, dims)
  metric <- c("euclidean", "euclidean", "squared-euclidean", "bray-curtis", "euclidean")[(i-1L) %% 5L+1L]
  D <- as.matrix(dist(C))
  if (metric == "squared-euclidean") D <- D^2
  if (metric == "bray-curtis") {
    A <- exp(C)
    for (a in seq_len(n)) for (b in seq_len(n))
      D[a,b] <- sum(abs(A[a,]-A[b,])) / sum(A[a,]+A[b,])
  }
  sites <- rep(c("zeta", "alpha", "mu", "beta", "eta")[seq_len(ns)], each=nt)
  surveys <- rep(seq_len(nt), ns)
  # Nonmonotone fractional times, shared between sites; times do not sort the axis.
  times <- sample(seq_len(nt) + runif(nt)/4)[surveys]
  p <- sample.int(n)
  # Guarantee an unsorted first-site axis, while retaining independent site orders.
  first <- which(sites[p] == sites[p[1]])
  if (!is.unsorted(surveys[p[first]])) p[first] <- rev(p[first])
  info <- list(nsites=ns, nsurveys=nt, dimensions=dims, metric=metric)
  record(sprintf("shuffle_%03d",i),D[p,p],sites[p],surveys[p],times[p],info)
  if (i <= 6L) record(sprintf("sorted_%03d",i),D,sites,surveys,times,info)
}
# Independent small edge cases, using the same public oracle and serializer.
ns <- 3L; nt <- 4L; n <- ns*nt
D <- as.matrix(dist(matrix(rnorm(n*5),n,5)))
sites <- rep(c("zeta","alpha","mu"),each=nt)
s <- rep(1:nt,ns)
p <- c(10,3,8,1,12,6,2,9,5,4,11,7)
edge <- function(id, surveys=s, times=s, keep=p) {
 record(id,D[keep,keep],sites[keep],surveys[keep],if(is.null(times)) NULL else times[keep])
}
edge("default_times", times=NULL)
edge("site_specific_surveys", s+rep(c(0,10,20),each=nt))
edge("duplicate_surveys", rep(c(1,1,2,3),ns))
edge("per_site_time_permutations", times=c(1:4,4:1,c(2,4,1,3)))
edge("repeated_times", times=rep(c(1,1,2,3),ns))
edge("constant_times", times=rep(1,n))
edge("unequal_time_multiplicities", times=c(1,1,2,3,1,2,2,3,1,2,3,3))
edge("unequal_lengths_shared_times", times=rep(c(1,1,2,3),ns), keep=p[p != 1])
edge("missing_survey", keep=p[p != 12])
edge("different_time_set", times=c(s[1:11],5))
edge("near_but_unequal_times", times=c(s[1:11],4+1e-9))
edge("fractional_surveys", surveys=s/10)
edge("label_surveys", surveys=letters[s])
edge("string_surveys", surveys=as.character(s))
# Repeated times change the number of rows contributing to each dispersion.
for (i in 1:8) {
  nt <- 8L; n <- ns*nt
  D <- as.matrix(dist(matrix(rnorm(n*5),n,5)))
  sites <- rep(c("zeta","alpha","mu"),each=nt)
  s <- rep(seq_len(nt),ns); p <- sample(n)
  edge(sprintf("repeated_time_stress_%02d",i), times=rep(c(1,1,1,1,2,2,3,4),ns))
}
write_json(list(metadata=list(R=R.version.string,
 ecotraj=as.character(packageVersion("ecotraj")),seed=seed,
 shuffled_cases=count, schema_version=1L),cases=cases),output,
 digits=I(17),auto_unbox=TRUE,na="null",null="null",pretty=TRUE)
cat("Wrote",length(cases),"cases to",output,"\n")
