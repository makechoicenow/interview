# 秒杀系统

# 搜索提示

# 短链接系统

# Instagram

# Dropbox

# Facebook Messager

# Twitter

# Youtube

# RateLimiter

## Scene

### 功能
+ Time Window Limit/ QPS limit
+ Level: UserId/API/USER+API
+ Cluster RateLimit
+ Type of thorttling
	+ Hard
	+ Soft
	+ Elastic
## Service
+ client
+ Api Server
+ Config center service
+ Rate Limiter
+ First Ask Rate Limiter, If allow, then ask Api Server

## Storage
+ mysql
+ redis

## Scale
+ Algorithm
	+ Fix Window
	+ Sliding Window
	+ Token Bucket
+ Cluser
	+ 设置单机容量
	+ Lua and Redis

# Twitter Search

## Scene

### 预估

## Service
+ Application Server
+ Storage Server
+ Index Server
+ Metadata service (?)

## Storage
+ Mysql
+ Hbase
+ redis

## Scale
+ 如何创建全局唯一的tweetID
+ Index Server Partition
	+ Based on words
	+ Based on twitte id

+ 数据冷热分离
+ 离线构建
+ 实时流同步


# Web Crawler

# Facebook Newsfeed

## Scene

### 功能
+ 发布
+ 内容可能不只是文字
+ 收到自己所有关注的推
+ 顺序不变
+ 2s以内完成

### 预估

## Service

### DataBase Desigin
+ User
+ Entity(page、group)
+ UserRelation
+ FeedItem
+ FeedMedia
+ Media

### HighLevelSystemDesin
+ Web Servers
+ Application Servers
+ Metadata database and cache
+ Post database and cache
+ Video and photo storage, and cache
+ Newfeed gereration service
+ Feed Rank service
+ Feed Notificatuon service

## Storage
+ Mysql
+ Redis
+ CephFS
+ CDN

## Scale

### 推拉结合
+ 大V, pull
+ 非大V, push
+ 推的时候，只推给在线的好友

### Data Parititon
+ Sharding posts and metadata , 参考twitter 
+ Sharding feed data
	+ UserId hash
	+ 长度只保留500
	+ growth and replication, Consistent Hashing

# Yelp
## scene
### 功能
+ 上传或者下家消费场所(Location)
+ 查找附近的消费场所
+ 评价消费场所(Review)
## Service
### Database design
+ Location
+ Review
### Grids
### QuadTree
### Service
+ Edit Location Service : Update Database and update QuadTree
+ Review Location Service
+ Aggregation Server
+ QuadTree Server
+ QuadTree Indexer

## Storage
+ Mysql
+ redis
+ QudaTreeStorage

## Scale



# Uber

# TicketMaster

# 模型训练系统

# KV存储

# RPC

