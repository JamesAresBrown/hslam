
ADD_EXECUTABLE(pmvs2 pmvs2.cc)

TARGET_LINK_LIBRARIES(pmvs2 pmvs_lib image_lib numeric_lib ${PMVS_LIBRARIES} nlopt tinycthread)

INSTALL(TARGETS pmvs2 DESTINATION bin/)
