import re
import os
import sys

regex_product = re.compile(r"(ht|f)tp(s?)\:\/\/item([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\,\'\/\\\+&amp;%\$#_]*)?", re.S)
regex_img = re.compile(r"(ht|f)tp(s?):\/\/(.+\/)+.+(\.(gif|png|jpg|jpeg|webp|svg|psd|bmp|tif))")
# regex_url = re.compile(r"(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\,\'\/\\\+&amp;%\$#_]*)?",re.S)
regex_url = re.compile(r"(ht|f)tp(s?)\:\/\/[^\s]+", re.S)
regex_vedio = re.compile(r"http[s]?:[^\s]*(video|mp4)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))*")

def del_spec(s):
    spec = ["</s>", "<!DOCTYPE html>", "<html>", "<head>", "<title>", "</head>", "<body>", "<h1>", "<h2>", "<h3>", "<h4>", "<h5>", "<h6>",
            "<p>", "<a>", "<strong>", "<em>", "<br>", "<span>", "<ul>", "<ol>", "<li>", "<table>", "<tr>", "<td>", "<th>", "<form>", "<input>",
            "<textarea>", "<select>", "<button>", "<img>", "<audio>", "<video>", "<style>", "<script>", "<div>", "<span>", "<hr>", "<s>", 
            "</div>", "</p>", "</a>", "</li>", "</pre>", "</section>", "</span>", "</h2>", "</ul>", "</button>", "</select>"]
    for sp in spec:
        s = s.replace(sp, ' ')
    return s


def msg_sub(msg):
    text = ''
    for i in range(len(msg)//10000+1):
        text += tok_sub(msg[i*10000:(i+1)*10000])
    return text
    

def tok_sub(text):
    text = re.sub(regex_product, "[商品]", text)
    text = re.sub(regex_img, "[图]", text)
    text = re.sub(regex_url, "[链接]", text)
    text = re.sub(regex_vedio, "[视频]", text)
    return text


if __name__ == "__main__":
    text = "买家发送：https://haohuo.jinritemai.com/ecommerce/trade/detail/index.html?id=3637771529904899034&origin_type=1337&slardar_report_id=7275235981141384740&channel_id=99514375927&channel_type=1\n机器人发送：您的眼镜顾问来啦\n买家发送：选度数\n买家发送：日抛\n人工发送：在的\n人工发送：比如我左眼300 右眼400，我就各自买一盒，然后各取一片即可组成一对~\r\n每一片都是带单独塑封哒，效期也是比较久，可以放心\n人工发送：亲，左右眼度数不同，需要买2盒，选择您需要的颜色/度数，加入购物车，在购物车选择需要的产品一起付款即可~。\r\n店铺产品1盒是1个度数，无法1盒发不同的度数，下单备注的订单，仓库是按照订单拍下的度数发货的哦。\n买家发送：可以申请退货吗？\n人工发送：可以\n"
    print(msg_sub(text))
