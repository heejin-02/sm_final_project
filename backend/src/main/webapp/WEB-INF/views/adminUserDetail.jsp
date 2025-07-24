<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
	<h2>회원 상세 정보</h2>
	<table border="1">
	    <tr><th>이름</th><td>${user.userName}</td></tr>
	    <tr><th>전화번호</th><td>${user.userPhone}</td></tr>
	    <tr><th>가입일</th><td>${user.joinedAt}</td></tr>
	</table>
	
	<a href="/admin">← 목록으로 돌아가기</a>

</body>
</html>