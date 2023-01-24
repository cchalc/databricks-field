# Databricks notebook source
# MAGIC %md
# MAGIC This notebook creates an init script (stored in DBFS) that should installs PDAL and major dependencies.

# COMMAND ----------

dbutils.fs.put(
  "/init/install-pdal-DBR9.1.sh",
"""#!/bin/bash
phycores=$(cat /proc/cpuinfo|grep -m 1 "cpu cores"|awk '{print $ 4;}')
add-apt-repository ppa:ubuntugis/ppa
apt-get update
apt-get install -y gdal-bin libgdal-dev gcc-multilib

git clone https://github.com/hobu/laz-perf.git
cd laz-perf
git checkout tags/2.1.0
mkdir build
cd build
cmake ..
make -j $phycores
make install
cd /databricks/driver

git clone https://github.com/PDAL/PDAL.git pdal
cd pdal
mkdir build
cd build
cmake   -G "Unix Makefiles"  \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DWITH_ICONV=ON \
        -DWITH_LASZIP=ON \
        -DWITH_LAZPERF=ON \
        -DWITH_LIBXML2=ON \
        -DBUILD_PLUGIN_PYTHON=ON \
        -DBUILD_PLUGIN_ICEBRIDGE=OFF \
        -DBUILD_PLUGIN_NITF=OFF \
        -DBUILD_PLUGIN_PGPOINTCLOUD=OFF \
        -DBUILD_PLUGIN_SQLITE=OFF \
        -DBUILD_PLUGIN_GREYHOUND=OFF \
        ..

make -j $phycores
sudo make install
/databricks/python3/bin/pip install PDAL
""", True)

# COMMAND ----------

# MAGIC %sh sh /dbfs/home/stuart@databricks.com/init/install-pdal-DBR9.1.sh

# COMMAND ----------

# MAGIC %fs ls /home/stuart@databricks.com/init
