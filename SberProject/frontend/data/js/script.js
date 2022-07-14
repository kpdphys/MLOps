function getSearchResults(url="http://SEARCH_HOST/search/") {
    const query = $('#query').val();
    $.ajax({
        url: url,
        method: "get",
        dataType: "json",
        contentType: "application/json",
        data: {q: query},
        success: function(data, status, xhr) {
            const body = JSON.parse(xhr.responseText);
            const result = JSON.parse(body)["search"];
            const request = {"query": query, "result": result, "link_pos": 0};
            console.log(body);
            postSearchResults(request);

            let html = [];
                if (result.length == 0) {
                    html.push("<b>По Вашему запросу ничего не обнаружено :(</b>");
                }
            for (let i = 0, length = result.length; i < length; i++) {
                let pos = i + 1;
                html.push(`<li> <a href="javascript:void(0)" class="link" id=${pos} uri=${result[i]["uri"]}><b>${result[i]["title"]}</b></a> </li>`);
            }
            document.querySelector('#result').innerHTML = html.join('');
        },
        error: function(xhr, status, exception) {
            console.error(xhr.responseText);
        }
    });
}

function postSearchResults(result, url="http://METRICS_HOST/searches/") {
    $.ajax({
        url: url,
        method: "post",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify(result),
        success: function(data, status, xhr) {
            const body = JSON.parse(xhr.responseText);
            $('.link').on('click', wrapper(url + body["data"]["id"]));
        },
        error: function(xhr, status, exception) {
            console.error(xhr.responseText);
        }
    });
}

function wrapper(put_url) {
    function on_click_function() {
        obj = $(this);
        $.ajax({
            url: put_url,
            method: "put",
            dataType: "json",
            data: JSON.stringify({link_pos: obj.attr("id")}),
            contentType: "application/json",
            success: function(data, status, xhr) {
                const json_obj = JSON.parse(xhr.responseText);
                console.log(json_obj);
                window.open(obj.attr("uri"), '_blank');
            },
            error: function(xhr, status, exception) {
                console.error(xhr.responseText);
                window.open(obj.attr("uri"), '_blank');
            }
        });
        document.querySelector('#result').innerHTML = "";
    }
    return on_click_function
}
