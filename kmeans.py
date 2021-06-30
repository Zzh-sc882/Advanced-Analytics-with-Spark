from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.clustering import KMeans, KMeansModel
import pyspark.ml.clustering as clus
import pyspark.ml.feature as ft
from pyspark.python.pyspark.shell import sqlContext
from pyspark.sql.functions import *
import random
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vector, Vectors
from pyspark.sql import SQLContext

# numericOnly
def clusterselect1(numericOnly, k):
    assembler = VectorAssembler(inputCols=numericOnly.columns[:38], outputCol="featureVector", handleInvalid="skip")
    kmeans = KMeans().setMaxIter(40).setTol(1.0e-5).setSeed(random.randint(2147483647, 9223372036854775807)).setK(k) \
        .setPredictionCol(
        "cluster").setFeaturesCol("featureVector")
    pipeline = Pipeline(stages=[assembler, kmeans])
    kmeansModel = pipeline.fit(numericOnly)
    pipelineModel = kmeansModel.stages[-1]
    sse = pipelineModel.computeCost(assembler.transform(numericOnly)) / numericOnly.count()
    re = str(k) + " , " + str(sse)
    return re


# 特征规范化  numericOnly
def clusterselect2(numericOnly, k):
    assembler = VectorAssembler(inputCols=numericOnly.columns[:38], outputCol="featureVector", handleInvalid="skip")
    scaler = StandardScaler(inputCol="featureVector", outputCol="scaledFeatureVector", withStd=True, withMean=False)
    kmeans = KMeans().setMaxIter(40).setTol(1.0e-5).setSeed(random.randint(2147483647, 9223372036854775807)).setK(
        k).setPredictionCol(
        "cluster").setFeaturesCol("scaledFeatureVector")
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    kmeansModel = pipeline.fit(numericOnly)
    pipelineModel = kmeansModel.stages[-1]
    sse = pipelineModel.computeCost(kmeansModel.transform(numericOnly)) / numericOnly.count()
    re = str(k) + " , " + str(sse)
    return re


# 类别型变量  data
def oneHotPipeline(inputCol):
    indexer = StringIndexer().setInputCol(inputCol).setOutputCol(inputCol + "_indexed")
    encoder = OneHotEncoder().setInputCol(inputCol + "_indexed").setOutputCol(inputCol + "_vec")
    pipeline = Pipeline(stages=[indexer, encoder])
    return pipeline, inputCol + "_vec"


def clusterselect3(data, k):
    protoTypeEncoder, protoTypeVecCol = oneHotPipeline("protocol_type")
    serviceEncoder, serviceVecCol = oneHotPipeline("service")
    flagEncoder, flagVecCol = oneHotPipeline("flag")
    assembleCols = set(data.columns) - {"label", "protocol_type", "service", "flag"}
    assembleCols.update({protoTypeVecCol, serviceVecCol, flagVecCol})
    assembler = VectorAssembler(inputCols=list(assembleCols), outputCol="featureVector")
    scaler = StandardScaler(inputCol="featureVector", outputCol="scaledFeatureVector", withStd=True, withMean=False)
    kmeans = KMeans().setMaxIter(40).setTol(1.0e-5).setSeed(random.randint(2147483647, 9223372036854775807)).setK(
        k).setPredictionCol("cluster").setFeaturesCol("scaledFeatureVector")
    pipeline = Pipeline(stages=[protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans])
    pipelineModel = pipeline.fit(data)
    kmeansModel = pipelineModel.stages[-1]
    sse = kmeansModel.computeCost(pipelineModel.transform(data)) / data.count()
    re = str(k) + " , " + str(sse)
    return re


# def entropy():
#     return
#
#
# def clusterselect4(data, k):
#     pipelineModel = fitPipeline4(data, k)
#     clusterLabel = pipelineModel.transform(data).select("cluster", "label")
#     weightedClusterEntropy = clusterLabel.groupByKey("cluster").count().collect()
#
#     return
#
#
def fitPipeline4(data, k):
    protoTypeEncoder, protoTypeVecCol = oneHotPipeline("protocol_type")
    serviceEncoder, serviceVecCol = oneHotPipeline("service")
    flagEncoder, flagVecCol = oneHotPipeline("flag")
    assembleCols = set(data.columns) - {"label", "protocol_type", "service", "flag"}
    assembleCols.update({protoTypeVecCol, serviceVecCol, flagVecCol})
    assembler = VectorAssembler(inputCols=list(assembleCols), outputCol="featureVector")
    scaler = StandardScaler(inputCol="featureVector", outputCol="scaledFeatureVector", withStd=True, withMean=False)
    kmeans = KMeans().setMaxIter(40).setTol(1.0e-5).setSeed(random.randint(2147483647, 9223372036854775807)).setK(
        k).setPredictionCol("cluster").setFeaturesCol("scaledFeatureVector")
    pipeline = Pipeline(stages=[protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans])
    pipelineModel = pipeline.fit(data)
    return pipelineModel


def buildAnomalyDetector(data):
    pipelineModel = fitPipeline4(data, 180)
    kMeansModel = pipelineModel.stages[-1]
    centroids = kMeansModel.clusterCenters()
    clustered = pipelineModel.transform(data)
    threshold = clustered.select("cluster", "scaledFeatureVector").rdd \
        .map(lambda x: Vectors.squared_distance(centroids[x[0]], x[1])).sortBy(lambda value: value,
                                                                               ascending=False).take(100)[-1]
    originalCols = data.columns
    anomalies = clustered.rdd.filter(lambda x: Vectors.squared_distance(centroids[x[-1]], x[-2]) >= threshold)
    dfanomalies = SQLContext.createDataFrame(anomalies)
    dfanomalies.first()


if __name__ == '__main__':
    spark = SparkSession.builder.config("spark.debug.maxToStringFields", "100").getOrCreate()
    df = spark.read.csv("file:///export/data/sparkdata/kddcup.data_10_percent_corrected",
                        header=False, inferSchema=True)
    data = df.toDF(
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label")
    data.groupBy("label").count().show()
    data.cache()
    numericOnly = data.drop("protocol_type", "service", "flag").cache()
    # k选择
    # result = []
    # for k in range(20, 201, 20):
    #     re = clusterselect1(numericOnly, k)
    #     result.append(re)
    # for i in result:
    #     print(i)
    # 特征规范化
    # result = []
    # for k in range(60, 271, 30):
    #     re = clusterselect2(numericOnly, k)
    #     result.append(re)
    # for i in result:
    #     print(i)
    # 类别型变量
    # result = []
    # for k in range(60, 271, 30):
    #     re = clusterselect3(data, k)
    #     result.append(re)
    # for i in result:
    #     print(i)
    # 利用标号的熵信息k选择
    # result = []
    # for k in range(20, 271, 30):
    #     re = clusterselect4(data, k)
    #     result.append(re)
    # for i in result:
    #     print(i)
    # pipelineModel = fitPipeline4(data, 180)
    # countByClusterLabel = pipelineModel.transform(data) \
    #     .select("cluster", "label").groupBy("cluster", "label") \
    #     .count().orderBy("cluster", "label")
    # countByClusterLabel.show()
    buildAnomalyDetector(data)
