CREATE TABLE `reddit-scraper.reddit.comments1`
AS
SELECT id,body,score,link_id FROM
(
 Select id,
        body,
        score,
        link_id,
        ROW_NUMBER() OVER (PARTITION BY link_id ORDER BY score DESC) AS comment_rank,
        FROM (
            SELECT * FROM 
            `fh-bigquery.reddit_comments.201*`
            WHERE link_id IN (SELECT id FROM `reddit-scraper.reddit.ids`)
            AND REGEXP_CONTAINS(_TABLE_SUFFIX, r"._..$")
            ORDER BY score DESC
            )
        WHERE 
        NOT LOWER(author)="automoderator"
        AND NOT body in ("[deleted]","[removed]")
        AND NOT LOWER(author) LIKE "%bot"

)
WHERE comment_rank <= 20
ORDER BY link_id