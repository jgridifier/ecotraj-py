#!/usr/bin/env Rscript
# Clean-room oracle: public ecotraj calls only, no implementation inspection.
suppressPackageStartupMessages(library(ecotraj))
library(jsonlite)
stopifnot(as.character(packageVersion('ecotraj')) == '1.2.2')
args <- commandArgs(FALSE)
script <- sub('^--file=', '', args[grepl('^--file=', args)])
root <- dirname(dirname(normalizePath(script)))
set.seed(5)
walk <- cumsum(rnorm(30))
set.seed(19)
walk2 <- cumsum(rnorm(12))
paths <- list(seed5_1d=matrix(walk,ncol=1), seed19_1d=matrix(walk2,ncol=1),
  seed5_horizontal=cbind(walk,0), seed19_diagonal=cbind(walk2,2*walk2),
  integer_reversals=matrix(c(0,3,1,4,-2,5),ncol=1),
  repeated_states=matrix(c(0,1,0,1,1,2),ncol=1),
  decimal_straight=matrix(c(0,.1,.3),ncol=1),
  decimal_reversal=matrix(c(0,.3,.1),ncol=1))
cases <- lapply(names(paths), function(id) {
  C <- paths[[id]]
  x <- defineTrajectories(dist(C),rep('path',nrow(C)),seq_len(nrow(C)))
  list(id=id,C=unname(C),d=unname(as.matrix(dist(C))),
       angles=unname(as.matrix(suppressWarnings(trajectoryAngles(x,all=TRUE,stats=FALSE)))),
       angles_noadd=unname(as.matrix(suppressWarnings(trajectoryAngles(x,all=TRUE,stats=FALSE,add=FALSE)))),
       directionality=unname(suppressWarnings(trajectoryDirectionality(x))),
       directionality_noadd=unname(suppressWarnings(trajectoryDirectionality(x,add=FALSE))))
})
write_json(list(metadata=list(R=R.version.string,ecotraj=as.character(packageVersion('ecotraj'))),cases=cases),
 file.path(root,'tests/fixtures/collinear_cases.json'),digits=I(17),auto_unbox=TRUE,na='null',pretty=TRUE)
