<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/fmt" prefix="fmt" %>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
<h1> 전체 회원 정보</h1>
<div>
	<button onclick="location.href='insert'" class="btn">회원 추가</button>
	<table>
		<tr>
			<th>회원번호</th>
			<th>이름</th>
			<th>전화번호</th>
			<th>아이디</th>			
			<th>대표농장이름</th>
			<th>대표농장주소</th>
			<th>대표농장번호</th>
		</tr>
		<c:forEach var="farm" items="${farmList}">
		    <tr>
		        <td>${farm.farmIdx}</td>
		        <td>${farm.farmName}</td>
		        <td>${farm.farmAddr}</td>
		        <td>${farm.farmPhone}</td>
		        <td>${farm.farmCrops}</td>
		        <td>${farm.farmArea}</td>
		        <td>${farm.createdAt}</td>
		        <td><img src="${farm.farmImg}" width="100"/></td>
		        <td>${farm.userPhone}</td>
		    </tr>
		</c:forEach>

	</table>
</div>
</body>
</html>