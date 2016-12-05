TOPDIR=..
include $(TOPDIR)/make.inc


#######################################################################
#  This is the makefile to create a library for LAPACK.
#  The files are organized as follows:
#       ALLAUX -- Auxiliary routines called from all precisions
#       ALLXAUX -- Auxiliary routines called from all precisions but
#                  only from routines using extra precision.
#       SCLAUX -- Auxiliary routines called from both REAL and COMPLEX
#       DZLAUX -- Auxiliary routines called from both DOUBLE PRECISION
#                 and COMPLEX*16
#       SLASRC -- Single precision real LAPACK routines
#       SXLASRC -- Single precision real LAPACK routines using extra
#                  precision.
#       CLASRC -- Single precision complex LAPACK routines
#       CXLASRC -- Single precision complex LAPACK routines using extra
#                  precision.
#       DLASRC -- Double precision real LAPACK routines
#       DXLASRC -- Double precision real LAPACK routines using extra
#                  precision.
#       ZLASRC -- Double precision complex LAPACK routines
#       ZXLASRC -- Double precision complex LAPACK routines using extra
#                  precision.
#
#  The library can be set up to include routines for any combination
#  of the four precisions.  To create or add to the library, enter make
#  followed by one or more of the precisions desired.  Some examples:
#       make single
#       make single complex
#       make single double complex complex16
#  Alternatively, the command
#       make
#  without any arguments creates a library of all four precisions.
#  The library is called
#       lapack.a
#  and is created at the next higher directory level.
#
#  To remove the object files after the library is created, enter
#       make clean
#  On some systems, you can force the source files to be recompiled by
#  entering (for example)
#       make single FRC=FRC
#
#  ***Note***
#  The functions lsame, second, dsecnd, slamch, and dlamch may have
#  to be installed before compiling the library.  Refer to the
#  installation guide, LAPACK Working Note 41, for instructions.
#
#######################################################################

ALLAUX = maxloc.o ilaenv.o ieeeck.o lsamen.o xerbla.o xerbla_array.o iparmq.o	\
    ilaprec.o ilatrans.o ilauplo.o iladiag.o chla_transtype.o \
    ../INSTALL/ilaver.o ../INSTALL/lsame.o

ALLXAUX =

DZLAUX = \
   dbdsdc.o \
   dbdsqr.o ddisna.o dlabad.o dlacpy.o dladiv.o dlae2.o  dlaebz.o \
   dlaed0.o dlaed1.o dlaed2.o dlaed3.o dlaed4.o dlaed5.o dlaed6.o \
   dlaed7.o dlaed8.o dlaed9.o dlaeda.o dlaev2.o dlagtf.o \
   dlagts.o dlamrg.o dlanst.o \
   dlapy2.o dlapy3.o dlarnv.o \
   dlarra.o dlarrb.o dlarrc.o dlarrd.o dlarre.o dlarrf.o dlarrj.o \
   dlarrk.o dlarrr.o dlaneg.o \
   dlartg.o dlaruv.o dlas2.o  dlascl.o \
   dlasd0.o dlasd1.o dlasd2.o dlasd3.o dlasd4.o dlasd5.o dlasd6.o \
   dlasd7.o dlasd8.o dlasda.o dlasdq.o dlasdt.o \
   dlaset.o dlasq1.o dlasq2.o dlasq3.o dlasq4.o dlasq5.o dlasq6.o \
   dlasr.o  dlasrt.o dlassq.o dlasv2.o dpttrf.o dstebz.o dstedc.o \
   dsteqr.o dsterf.o dlaisnan.o disnan.o \
   ../INSTALL/dlamch.o ../INSTALL/dsecnd.o

DLASRC = \
   dgbbrd.o dgbcon.o dgbequ.o dgbrfs.o dgbsv.o  \
   dgbsvx.o dgbtf2.o dgbtrf.o dgbtrs.o dgebak.o dgebal.o dgebd2.o \
   dgebrd.o dgecon.o dgeequ.o dgees.o  dgeesx.o dgeev.o  dgeevx.o \
   dgegs.o  dgegv.o  dgehd2.o dgehrd.o dgelq2.o dgelqf.o \
   dgels.o  dgelsd.o dgelss.o dgelsx.o dgelsy.o dgeql2.o dgeqlf.o \
   dgeqp3.o dgeqpf.o dgeqr2.o dgeqrf.o dgerfs.o dgerq2.o dgerqf.o \
   dgesc2.o dgesdd.o dgesv.o  dgesvd.o dgesvx.o dgetc2.o dgetf2.o \
   dgetrf.o dgetri.o \
   dgetrs.o dggbak.o dggbal.o dgges.o  dggesx.o dggev.o  dggevx.o \
   dggglm.o dgghrd.o dgglse.o dggqrf.o \
   dggrqf.o dggsvd.o dggsvp.o dgtcon.o dgtrfs.o dgtsv.o  \
   dgtsvx.o dgttrf.o dgttrs.o dgtts2.o dhgeqz.o \
   dhsein.o dhseqr.o dlabrd.o dlacon.o dlacn2.o \
   dlaein.o dlaexc.o dlag2.o  dlags2.o dlagtm.o dlagv2.o dlahqr.o \
   dlahrd.o dlahr2.o dlaic1.o dlaln2.o dlals0.o dlalsa.o dlalsd.o \
   dlangb.o dlange.o dlangt.o dlanhs.o dlansb.o dlansp.o \
   dlansy.o dlantb.o dlantp.o dlantr.o dlanv2.o \
   dlapll.o dlapmt.o \
   dlaqgb.o dlaqge.o dlaqp2.o dlaqps.o dlaqsb.o dlaqsp.o dlaqsy.o \
   dlaqr0.o dlaqr1.o dlaqr2.o dlaqr3.o dlaqr4.o dlaqr5.o \
   dlaqtr.o dlar1v.o dlar2v.o iladlr.o iladlc.o \
   dlarf.o  dlarfb.o dlarfg.o dlarft.o dlarfx.o dlargv.o \
   dlarrv.o dlartv.o dlarfp.o \
   dlarz.o  dlarzb.o dlarzt.o dlaswp.o dlasy2.o dlasyf.o \
   dlatbs.o dlatdf.o dlatps.o dlatrd.o dlatrs.o dlatrz.o dlatzm.o dlauu2.o \
   dlauum.o dopgtr.o dopmtr.o dorg2l.o dorg2r.o \
   dorgbr.o dorghr.o dorgl2.o dorglq.o dorgql.o dorgqr.o dorgr2.o \
   dorgrq.o dorgtr.o dorm2l.o dorm2r.o \
   dormbr.o dormhr.o dorml2.o dormlq.o dormql.o dormqr.o dormr2.o \
   dormr3.o dormrq.o dormrz.o dormtr.o dpbcon.o dpbequ.o dpbrfs.o \
   dpbstf.o dpbsv.o  dpbsvx.o \
   dpbtf2.o dpbtrf.o dpbtrs.o dpocon.o dpoequ.o dporfs.o dposv.o  \
   dposvx.o dpotf2.o dpotrf.o dpotri.o dpotrs.o dpstrf.o dpstf2.o \
   dppcon.o dppequ.o \
   dpprfs.o dppsv.o  dppsvx.o dpptrf.o dpptri.o dpptrs.o dptcon.o \
   dpteqr.o dptrfs.o dptsv.o  dptsvx.o dpttrs.o dptts2.o drscl.o  \
   dsbev.o  dsbevd.o dsbevx.o dsbgst.o dsbgv.o  dsbgvd.o dsbgvx.o \
   dsbtrd.o  dspcon.o dspev.o  dspevd.o dspevx.o dspgst.o \
   dspgv.o  dspgvd.o dspgvx.o dsprfs.o dspsv.o  dspsvx.o dsptrd.o \
   dsptrf.o dsptri.o dsptrs.o dstegr.o dstein.o dstev.o  dstevd.o dstevr.o \
   dstevx.o dsycon.o dsyev.o  dsyevd.o dsyevr.o \
   dsyevx.o dsygs2.o dsygst.o dsygv.o  dsygvd.o dsygvx.o dsyrfs.o \
   dsysv.o  dsysvx.o \
   dsytd2.o dsytf2.o dsytrd.o dsytrf.o dsytri.o dsytrs.o dtbcon.o \
   dtbrfs.o dtbtrs.o dtgevc.o dtgex2.o dtgexc.o dtgsen.o \
   dtgsja.o dtgsna.o dtgsy2.o dtgsyl.o dtpcon.o dtprfs.o dtptri.o \
   dtptrs.o \
   dtrcon.o dtrevc.o dtrexc.o dtrrfs.o dtrsen.o dtrsna.o dtrsyl.o \
   dtrti2.o dtrtri.o dtrtrs.o dtzrqf.o dtzrzf.o dstemr.o \
   dsgesv.o dsposv.o dlag2s.o slag2d.o dlat2s.o \
   dlansf.o dpftrf.o dpftri.o dpftrs.o dsfrk.o dtfsm.o dtftri.o dtfttp.o \
   dtfttr.o dtpttf.o dtpttr.o dtrttf.o dtrttp.o \
   dgejsv.o  dgesvj.o  dgsvj0.o  dgsvj1.o \
   dgeequb.o dsyequb.o dpoequb.o dgbequb.o

DXLASRC = dgesvxx.o dgerfsx.o dla_gerfsx_extended.o dla_geamv.o		\
   dla_gercond.o dla_rpvgrw.o dsysvxx.o dsyrfsx.o			\
   dla_syrfsx_extended.o dla_syamv.o dla_syrcond.o dla_syrpvgrw.o	\
   dposvxx.o dporfsx.o dla_porfsx_extended.o dla_porcond.o		\
   dla_porpvgrw.o dgbsvxx.o dgbrfsx.o dla_gbrfsx_extended.o		\
   dla_gbamv.o dla_gbrcond.o dla_gbrpvgrw.o dla_lin_berr.o dlarscl2.o	\
   dlascl2.o dla_wwaddw.o

all: ../$(LAPACKLIB)

ifdef USEXBLAS
ALLXOBJ=$(DXLASRC) $(ALLXAUX)
endif

ALLOBJ=$(DLASRC) $(DZLAUX)	\
	$(ALLAUX)

../$(LAPACKLIB): $(ALLOBJ) $(ALLXOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ) $(ALLXOBJ)
	$(RANLIB) $@

double: $(DLASRC) $(ALLAUX) $(DZLAUX)
	$(ARCH) $(ARCHFLAGS) ../$(LAPACKLIB) $(DLASRC) $(ALLAUX) \
	$(DZLAUX)
	$(RANLIB) ../$(LAPACKLIB)

$(ALLAUX): $(FRC)
$(DZLAUX): $(FRC)
$(DLASRC): $(FRC)
ifdef USEXBLAS
$(ALLXAUX): $(FRC)
$(DXLASRC): $(FRC)
endif

FRC:
	@FRC=$(FRC)

clean:
	rm -f *.o

.c.o: 
	$(CC) $(CFLAGS) -c $<

dlaruv.o: dlaruv.c ; $(CC) $(NOOPT)  -c $< -o $@
dla_wwaddw.o: dla_wwaddw.c ; $(CC) $(NOOPT)  -c $< -o $@

