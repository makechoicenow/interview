# 秒杀系统

# 常见的几板斧
## 负载均衡
## 一致性哈希
## 热点缓存：二八定律+LRU
## 多副本
## 消息队列
## 冷热数据分离
## 分布式唯一ID的生成

# 搜索提示

# 短链接系统

# Dropbox

# Facebook Messager

#  Instagram or Twtter

## Storage

### Database Choose
+ Mysql
+ Hdfs
+ Hbase
+ Redis

### SQL Database Desgin
+ Tweet
+ User
+ UserRelation
+ Favorite

## Scale

### TweetID 与 分片
+ UserId
  + Hot User ?
  + 大V，关系链很长？
  + 解决办法：一致性哈希
+ TweetId
  + query all partition and higher latency
  + 解决办法：缓存
+ Tweet Creation Time
  + 优点：query top tweets quickly and only query a very small set of servers
  + 缺点：traffic not distributed
+ Combine By TweetId and Tweet Creation Time
  + 为任何给定用户创建好友动态最重要的要求之一就是从该用户关注的所有人那里获取最新的照片。为此，我们需要一种机制来对照片在创建时进行排序。为了有效地做到这一点，我们可以将照片创建时间作为PhotoID的一部分。因为我们在PhotoID上有一个主索引，所以很快就能找到最新的PhotoID。我们可以使用时间来记录。假设PhotoID有两部分，第一部分将表示epoch时间，而第二部分将是一个自动递增的序列。
  + Writing: don't have any secondary index(Creation time) will reduce our write latency
  + Reading: don't need to filter on creation-time as our primary key has epoch time in it




# Youtube

## Scene

### Functional Requirements
+ upload
+ view
+ share
+ comment
+ like/dislike/save
## Service
### HighLevelSystemDesin
+ Upload Video Service
+ Processing Queue 处理队列：每个上载的视频将被推送到处理队列，以便稍后进行出队以进行编码，缩略图生成和存储。
+ Encoder 编码器：将每个上传的视频编码为多种格式。
+ Thumbnails generator 缩略图生成器：为每个视频生成一些缩略图。
+ Video and Thumbnail storage 视频和缩略图存储：将视频和缩略图文件存储在某些分布式文件存储中。
+ User Database 用户数据库：用于存储用户信息，例如姓名，电子邮件，地址等。
+ Video metadata storage 视频元数据存储：一个元数据数据库，用于存储有关视频的所有信息，例如标题，系统中的文件路径，上载用户，总观看次数，喜欢，不喜欢等。它还将用于存储所有视频评论。

## Storage
### Database Design
+ Mysql
	+ User
	+ VideoMeta
	+ VideoComment
+ HDFS
	+ Video

## Scale
### 视频存储多副本，一写多读。
### 缩略图服务流量很多，可以采用BigTable+Cache
### 元数据分片：基于UserId会导致数据分布不均衡，基于VideoId分片，辅以热点数据缓存。
### 重复数据删除，指纹、特征抽取等
### 引入CDN

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

### DataBase Desigin
+ Mysql
	+ User
	+ Entity(page、group)
	+ UserRelation
	+ FeedItem
	+ FeedMedia
	+ Media
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

